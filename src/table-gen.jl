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

  set_col_measures(grouped_df::AbstractDataFrame) = begin
    reduce(DataFrame(), cols) do acc, c
      acc[c] = col_measures(grouped_df, c)
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


function mk_contigency_tbl(df::DataFrame, index_col::Symbol, header_col::Symbol)
  get_categories(col::Symbol) = unique(dropna(df[col]))

  headers::AbstractVector = get_categories(header_col)
  indexes::AbstractVector = get_categories(index_col)

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


function fisher_exact(df::DataFrame, index_col::Symbol, header_col::Symbol)

  con_table::Matrix{Int64} = begin
    ret::DataFrame = mk_contigency_tbl(df, index_col, header_col)
    Matrix(ret[:, 2:end])
  end

  rcall(symbol("fisher.test"), con_table)
end


@memoize is_categorical(col::Symbol) = in(col, [:race, :sex])


function calcanova_dx(col::Symbol, df::DataFrame=raw())
  da::DataArray = df[col]

  dxdata(dx::Symbol) = DataGroup(
    permute_na_floatsafe(da[df[:dx] .== dx]),
    dx)
  calcanova(map(dxdata, [:nc, :mci, :ad])...)
end


function calc_omnibus_dx(col::Symbol, df::DataFrame=raw())
  da::DataArray = df[col]

  f_exact() = fisher_exact(df, :dx, col)[symbol("p.value")][1]

  is_categorical(col) ? f_exact() :
    calcanova_dx(col, df).resultsInfo[1, :PValue]
end


function calc_indi_comparisons_dx(col::Symbol, df::DataFrame=raw())
  if is_categorical(col)
    indis::DataFrame = tukey(calcanova_dx)
    ix_to_dx = Dict(1 => :nc, 2 => :mci, 3 => :ad)
  end
end


convert_race(df::DataFrame) = [i == "B" ? 1 : 0 for i in df[:race]]


function get_pvalues(df::DataFrame=raw(), cols::Vector{Symbol}=demo_cols())
  df2::DataFrame = begin
    ret::DataFrame = copy(df[:, [:dx; cols]])
    in(:race, names(ret)) && (ret[:race] = convert_race(ret))
    ret
  end

  pval(c::Symbol) = calc_omnibus_dx(c, df2)

  pvalsraw::Dict{Symbol, Float64} = [c => pval(c) for c in cols]
  Dict(
    zip(keys(pvalsraw),
        padjust(collect(values(pvalsraw)), BenjaminiHochberg))
    )
end


function get_different_cols(df::DataFrame=raw(), cols::Vector{Symbol}=demo_cols(); alpha=.05)
  filter( (c, p) -> p < alpha, get_pvalues(df, cols))
end


function get_differents_df(df::DataFrame=raw(),
                           cols::Vector{Symbol}=demo_cols();
                           normalize::Bool=false)
  different_cols::Vector{Symbol} = collect(keys(get_different_cols()))

  input = copy(df[[:dx; different_cols]])
  for c in different_cols
    tmp::DataArray{Float64} = permute_na_floatsafe(input[c])
    if normalize
      na_ixs::Vector{Bool} = [isna(i)::Bool for i in length(input[c])]

      tmp[!na_ixs] = normalize_variables(input[c])
      tmp[na_ixs] = 0.
    end
    input[c] = tmp
  end

  stack(input, different_cols)
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


function rank_msg(items::AbstractVector{SimpleAnova.Label}, ranks::AbstractVector{Int64})
  sorted_ranks::Vector{Int64} = sort(unique(ranks), rev=true)
  comma_join(rank::Int64) = join(
    items[ranks .== rank],
    ", ")
  join(map(comma_join, sorted_ranks), " > ")
end


rank_msg(df::DataFrame) = rank_msg(df[:item], df[:rank])


function get_table()
  data_cols::Vector{Symbol} = names(group_by_dx())[2:end]

  ret = DataFrame()
  ret[:measure] = data_cols

  for r in 1:size(group_by_dx(), 1)
    dx::Symbol = group_by_dx()[r, :dx]
    measures::Vector{AbstractString} = Array(group_by_dx()[r, data_cols])[:]
    ret[dx] = measures
  end


  different_cols_msg::Vector{AbstractString} = begin
    different_cols_and_pvalues::Dict{Symbol, Float64} = get_different_cols()
    different_cols::Set{Symbol} = Set(keys(different_cols_and_pvalues))

    gen_msg(col::Symbol) = begin
      isSig = in(col, different_cols)
      if isSig
        pvalue::AbstractString = fmt_float(different_cols_and_pvalues[col])
        if is_categorical(col)
          "$pvalue, Fisher-Exact"
        else
          tukey_res::DataFrame = tukey(calcanova_dx(col))
          tukey_rank::DataFrame = rank(tukey_res)
          rank_msg(tukey_rank)
        end
      else
        "ns"
      end
    end

    map(gen_msg, data_cols)
  end


  header_maps::Dict{Symbol, Symbol} = begin
    count_msg::Dict{Symbol, AbstractString} = [
      dx => "(n=$(sum(raw()[:dx] .== dx)))" for dx in [:nc, :mci, :ad]]
    Dict(:nc => symbol("NC $(count_msg[:nc])"),
         :mci => symbol("MCI $(count_msg[:mci])"),
         :ad => symbol("AD $(count_msg[:ad])"),
         )
  end

  rename!(ret, header_maps)

  ret[symbol("Significance")] = different_cols_msg

  ret
end
