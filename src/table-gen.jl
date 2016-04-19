using DataFrames
using Formatting
using Lazy
using Memoize
using PValueAdjust
using RCall
using SimpleAnova

include("casl_lmm_fit.jl")

function permute_na(da::DataArray, fn::Function=mean)
  na_ixs::Vector{Bool} = map(isna, da)
  da[na_ixs] = fn(dropna(da))
  da
end


function permute_na_floatsafe{T}(da::AbstractVector{T}, fn::Function=mean)
  dfloat::AbstractVector{Float64} = @switch T begin
    Float64; da;
    convert2float(da)
  end

  Array(permute_na(dfloat, fn))
end

fmt_float(r::Float64) = ( abs(r)>1e3 || abs(r)<1e-3 ) ? format("{:.3e}", r) : format("{:.3f}", r)

function col_measures(df::AbstractDataFrame, col::Symbol)

  item_counts(label_fn) = begin
    cts = [label_fn(c, count(i -> i == c, df[col]))
           for c in unique(df[col])]
    join(sort(cts), ", ")
  end

  if col == :sex
    item_counts( (c, ct) -> c == 1 ? "M: $ct" : "F: $ct")
  elseif col == :race
    item_counts( (c, ct) -> "$c: $ct")
  else
    da::DataArray = df[col]
    isa(da, AbstractVector{Float64}) || (da = convert2float(da))

    mn, std = mean_and_std(Array(permute_na(da, mean)))
    "$(fmt_float(mn)) ($(fmt_float(std)))"
  end
end


demo_cols() = Symbol[symbol(n) for n in readcsv("$data_dir/input/demographics_cols.csv")]


function merge_aami_into_nc!(df::DataFrame)
  df[df[:dx] .== :aami, :dx] = :nc
  df
end


merge_aami_into_nc(df::DataFrame) = merge_aami_into_nc!(copy(df))


function group_by_dx(df::DataFrame=raw();
                     cols::AbstractVector{Symbol}=demo_cols())

  set_col_measures(sub_df::SubDataFrame) = begin
    reduce(DataFrame(), cols) do acc, c
      acc[c] = col_measures(sub_df, c)
      acc
    end
  end

  by(df, [:dx], set_col_measures)
end


save_group_by_dx(df::DataFrame,
                 f::AbstractString="$data_dir/step1/group_by_dx.csv") = writetable(f, df)


immutable Count
  c::Int64
end


true_fn(args...) = true


function mk_contigency_tbl(df::DataFrame,
                           index_col::Symbol,
                           header_col::Symbol,
                           index_filter::Function=true_fn)
  get_categories(col::Symbol) = unique(dropna(df[col]))

  headers::AbstractVector = get_categories(header_col)
  indexes::AbstractVector = filter(index_filter, get_categories(index_col))

  ret = DataFrame()
  ret[index_col] = indexes

  na_ixs(col::Symbol) = Bool[isna(i) for i in df[col]] #map wont state type bool
  not_nas::Vector{Bool} = !(na_ixs(header_col) | na_ixs(index_col))

  for h in headers
    data = zeros(Int64, length(indexes))
    for (ix::Int64, i) in enumerate(indexes)
      passing::Vector{Bool} = (df[header_col] .== h) & (df[index_col] .== i) & not_nas
      data[ix] = count(identity, passing)
    end
    ret[symbol(h)] = data
  end

  ret
end


function calc_individuals_categorical(header_col::Symbol,
                                      df::DataFrame=raw(),
                                      index_col::Symbol=:dx;
                                      valid_indexes::AbstractVector{Symbol}=[:nc, :mci, :ad]
                                      )
  lefts = Symbol[]
  rights = Symbol[]
  ps = Float64[]
  num_ixs::Int64 = length(valid_indexes)

  valid_indexes = unique(valid_indexes)

  for left_ix = 1:num_ixs
    for right_ix = (left_ix + 1):num_ixs
      left::Symbol, right::Symbol = valid_indexes[[left_ix, right_ix]]
      index_filter(icol::Symbol) = (icol == left) || (icol == right)
      omn_res = calc_omnibus_categorical(df, index_col, header_col, index_filter)
      p::Float64 = omn_res[symbol("p.value")][1]
      ps = [ps; p]
      lefts = [lefts; left]
      rights = [rights; right]
    end
  end

  ret = DataFrame(left=lefts, right=rights, pval=ps)
end


function calc_omnibus_categorical(df::DataFrame,
                                  index_col::Symbol,
                                  header_col::Symbol,
                                  index_filter::Function=true_fn)

  con_table::Matrix{Int64} = begin
    ret::DataFrame = mk_contigency_tbl(df, index_col, header_col, index_filter)
    Matrix(ret[:, 2:end])
  end

  rcall(symbol("fisher.test"), con_table)
end


@memoize is_categorical(col::Symbol) = in(col, [:race, :sex])


function calc_omnibus_continuous(x_col::Symbol, df::DataFrame=raw();
                      y_col::Symbol=:dx,
                      valid_indexes::Vector{Symbol}=[:nc, :mci, :ad])
  da::DataArray = df[x_col]

  classdata(class::Symbol) = SimpleAnova.DataGroup(
    permute_na_floatsafe(da[df[y_col] .== class]),
    class)

  calcanova(map(classdata, valid_indexes)...)
end


function calc_omnibus(col::Symbol, df::DataFrame=raw())
  da::DataArray = df[col]

  f_exact() = calc_omnibus_categorical(df, :dx, col)[symbol("p.value")][1]

  is_categorical(col) ? f_exact() :
    calc_omnibus_continuous(col, df).resultsInfo[1, :PValue]
end


convert_race(df::DataFrame) = [i == "B" ? 1 : 0 for i in df[:race]]


function get_pvalues(df::DataFrame=raw(), cols::Vector{Symbol}=demo_cols())
  df2::DataFrame = begin
    ret::DataFrame = copy(df[:, [:dx; cols]])
    in(:race, names(ret)) && (ret[:race] = convert_race(ret))
    ret
  end

  pval(c::Symbol) = calc_omnibus(c, df2)

  pvalsraw::Dict{Symbol, Float64} = [c => pval(c) for c in cols]
  Dict(
    zip(keys(pvalsraw),
        padjust(collect(values(pvalsraw)), BenjaminiHochberg))
    )
end


function get_different_cols(df::DataFrame=raw(), cols::Vector{Symbol}=demo_cols(); alpha=.05)
  filter( (c, p) -> p < alpha, get_pvalues(df, cols))
end


function rank{T}(lefts::AbstractVector{T}, rights::AbstractVector{T}, diffs::AbstractVector{Bool})
  items::Vector{T} = union(lefts, rights)
  ranks::Vector{Int64} = zeros(Int64, length(items))
  ret = DataFrame(item=items, rank=ranks)

  @assert length(lefts) == length(rights) == length(diffs)

  get_rank(item::T) = ret[ret[:item] .== item, :rank][1]
  update_rank!(item::T, new_rank::Int64) = begin
    curr_rank::Int64 = get_rank(item)
    (curr_rank < new_rank) && (ret[ret[:item] .== item, :rank] = new_rank)
  end

  for item::T in items
    sig_parents::AbstractVector{T} = lefts[(rights .== item) & diffs]
    for s::T in sig_parents
      update_rank!(s, get_rank(item) + 1)
    end
  end

  ret
end


rank(df::DataFrame, alpha=.05) = rank(df[:left], df[:right], df[:pval] .< alpha)


function rank_msg{T <: Union{Symbol, ASCIIString}}(
    items::AbstractVector{T},
    ranks::AbstractVector{Int64},
    diff_join::AbstractString,
    equal_join::AbstractString=", ")

  sorted_ranks::Vector{Int64} = sort(unique(ranks), rev=true)
  comma_join(rank::Int64) = join(
    items[ranks .== rank],
    equal_join)
  join(map(comma_join, sorted_ranks), diff_join)
end


rank_msg(df::DataFrame, diff_join::AbstractString) = rank_msg(df[:item], df[:rank], diff_join)


function add_controls_gt_60(df::DataFrame = raw())
  orig::DataFrame = df
  nc_gt_60::Vector{Bool} = (orig[:age] .>= 60) & (orig[:dx] .== :nc)
  gt_60::DataFrame = orig[nc_gt_60, :]
  gt_60[:dx] = :nc_gt_60
  vcat(orig, gt_60)
end


function get_table()
  df::DataFrame = add_controls_gt_60()
  grouped_df::DataFrame = group_by_dx(df)

  data_cols::Vector{Symbol} = names(grouped_df)[2:end]

  ret = DataFrame()
  ret[:measure] = data_cols

  dxs::AbstractVector{Symbol} = collect(unique(df[:dx]))

  for r in 1:size(grouped_df, 1)
    dx::Symbol = grouped_df[r, :dx]
    measures::Vector{AbstractString} = Array(grouped_df[r, data_cols])[:]
    ret[dx] = measures
  end

  different_cols_msg::Vector{AbstractString} = begin
    different_cols_and_pvalues::Dict{Symbol, Float64} = get_different_cols(df)
    different_cols::Set{Symbol} = Set(keys(different_cols_and_pvalues))

    gen_msg(col::Symbol) = begin
      isSig = in(col, different_cols)
      if isSig
        indis::DataFrame, diff_join::AbstractString = is_categorical(col) ?
          (
            calc_individuals_categorical(col, df, valid_indexes=dxs),
            " not eq. ") :
          (
            tukey(calc_omnibus_continuous(col, df, valid_indexes=dxs)),
            " > ")

        ranks::DataFrame = rank(indis)
        rank_msg(ranks, diff_join)
      else
        "ns"
      end
    end

    map(gen_msg, data_cols)
  end


  header_maps::Dict{Symbol, Symbol} = begin
    count_msg::Dict{Symbol, AbstractString} = [
      dx => "(n=$(sum(df[:dx] .== dx)))" for dx in dxs]
    count_kv(c::Symbol, msg::AbstractString) = c => symbol("$msg $(count_msg[c])")
    Dict(count_kv(:nc, "NC"),
         count_kv(:mci, "MCI"),
         count_kv(:ad, "AD"),
         count_kv(:nc_gt_60, "NC 60+")
         )
  end

  rename!(ret, header_maps)

  ret[:Significance] = different_cols_msg

  ret
end
