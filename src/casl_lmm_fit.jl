using DataFrames
using Formatting
using Lazy
using Memoize
using MixedModels
using MultivariateStats # , PyPlot
using PValueAdjust
using SimpleAnova

srand(42)


data_dir  = "../data"

inputtable(f::AbstractString, separator='\t') = readtable("$data_dir/input/$f", separator=separator)

function convert2type{T}(da::DataArray, instance::T)
  na_ixs = [isna(i)::Bool for i in da]
  ret::DataArray{T} = @data([isna(i) ? instance :
                               convert(T, i) for i in da ])
  ret[na_ixs] = NA
  ret
end

convert2symbols(da::DataArray) = convert2type(da, :a)

convert2float(da::DataArray) = convert2type(da, 0.)

string2formula(s::AbstractString) = eval(parse(s))


function combine_main_effects(master_me;add_var=true)
    if add_var
        [ unique(push!(c,"fci")) for c in master_me ]
    else
        [ c for c in master_me ]
    end
end

function list_possible_interactions(me_list)
    interactions = [ string(c[1], " & ", c[2]) for c in combinations(me_list,2) ]
    all_combos = []
    for i in 0:length(interactions)
        append!(all_combos,[ join(c," + ") for c in combinations(interactions,i) ])
    end
    return all_combos
end

function concatenate_all_effects(me,base,inter)
    me = join(me," + ")
    base = join(base," + ")
    formula = "fci_fu ~ 1 + (1|subject)"
    if length(me) > 0 && length(base) > 0 && length(inter) > 0
        formula = "fci_fu ~ $me + $base + $inter + (1|subject)"
    end
    if length(me) == 0 && length(base) > 0
        formula = "fci_fu ~ $base + (1|subject)"
    elseif length(base) == 0 && length(me) > 0 && length(inter) > 0
        formula = "fci_fu ~ $me + $inter + (1|subject)"
    elseif length(base) == 0 && length(me) > 0 && length(inter) == 0
        formula = "fci_fu ~ $me + (1|subject)"
    end
    return formula
end

function list_all_formulae(me_list,base_list;test_interactions=true)
    mes = combine_main_effects(me_list,add_var=false)
    all_formulae = Set()
    for me in mes
        if test_interactions
            interacts = list_possible_interactions(me)
            for interaction in interacts
                f = concatenate_all_effects(me,base_list,interaction)
                push!(all_formulae,f)
            end
        else
            f = concatenate_all_effects(me,base_list,"")
            push!(all_formulae,f)
        end
    end
    return all_formulae
end


function extract_fx_from_formula(f)
    bitz = split(f," ~ ")
    bitz = split(bitz[2]," + ")
    bitz = map(s -> convert(Any,strip(s)),bitz)
    bitz = filter(y -> y != "(1|subject)",bitz)
    if in("ad",bitz) && in("mci",bitz)
        bitz = filter(x -> !in(x,["ad","mci"]),bitz)
        push!(bitz,"ad + mci")
    end
    return bitz
end

function normalize_variables(df, varnames)
  for varname in varnames
    mu, sd = mean_and_std(dropna(df[varname]))
    if sd != 0
      df[varname] = (df[varname]-mu)/sd
    end
  end

  df
end

function convert2symbols!(df::DataFrame, cols::Vector{Symbol})
  for c in cols
    df[c] = convert2symbols(df[c])
  end
end

function AICc(m,df)
    return (AIC(m) + (2npar(m) * (npar(m)+1))/(size(df,1)-npar(m)-1))
end

# New function to fit all regressions adding combinations of new effects
# to a base model.
function all_regressions(interactables,base_model,df;test_interactions=true)
    count = 1
    best_formula = "fci_fu ~ 1 + (1|subject)"
    best_model = fit!(lmm(string2formula(best_formula),df))
    best_AIC = AICc(best_model,df)
    if length(interactables) > 6
        lim = 6
    else
        lim = length(interactables)
    end
    for firsty in 0:lim
        main_effects = combinations(interactables,firsty)
        formulae = list_all_formulae(main_effects,base_model,test_interactions=test_interactions)
        for f in formulae
            m = fit!(lmm(string2formula(f),df))
            aic = AICc(m,df)
            if aic < best_AIC && isfinite(aic) && !isnan(aic)
                best_AIC = aic
                best_model = m
                best_formula = f
                println(best_formula)
            end
            m = 0
            count += 1
        end
        println("Count: $count")
        println("Current corrected AIC: $best_AIC")
        gc()
    end
    return (best_AIC,best_model,best_formula)
end

function build_all_models(casl,gm;n_me = ["fci", "atn", "exe", "mem", "sem", "time_elapsed"])
    all_models = Dict()
    # I.
    # Single domain models
    d_me = [ "age", "sex", "edu", "ad + mci" ]
    all_models["model_d"] = all_regressions(d_me,[],casl,test_interactions=false)

    # n_me = [  ]
    all_models["model_n"] = all_regressions(n_me,[],casl)

    g_me = [ string("pca_",i) for i in 1:19 ]
    all_models["model_g"] = all_regressions(g_me,[],gm,test_interactions=false)

    # II.
    # Two-domain all_models
    best_d = extract_fx_from_formula(all_models["model_d"][3])
    println("best_d: $best_d")
    all_models["model_dn"] = all_regressions(n_me,best_d,casl)
    all_models["model_dg"] = all_regressions(g_me,best_d,gm,test_interactions=false)

    best_n = extract_fx_from_formula(all_models["model_n"][3])
    println("best_n: $best_n")
    all_models["model_nd"] = all_regressions(d_me,best_n,casl,test_interactions=false)
    all_models["model_ng"] = all_regressions(g_me,best_n,gm,test_interactions=false)

    best_g = extract_fx_from_formula(all_models["model_g"][3])
    println("best_g: $best_g")
    all_models["model_gn"] = all_regressions(n_me,best_g,gm)
    all_models["model_gd"] = all_regressions(d_me,best_g,gm,test_interactions=false)

    # III.
    # Three-domain models
    best_dn = extract_fx_from_formula(all_models["model_dn"][3])
    all_models["model_dng"] = all_regressions(g_me,best_dn,gm,test_interactions=false)

    best_dg = extract_fx_from_formula(all_models["model_dg"][3])
    all_models["model_dgn"] = all_regressions(n_me,best_dg,gm)

    best_nd = extract_fx_from_formula(all_models["model_nd"][3])
    all_models["model_ndg"] = all_regressions(g_me,best_nd,gm,test_interactions=false)

    best_ng = extract_fx_from_formula(all_models["model_ng"][3])
    all_models["model_ngd"] = all_regressions(d_me,best_ng,gm,test_interactions=false)

    best_gn = extract_fx_from_formula(all_models["model_gn"][3])
    all_models["model_gnd"] = all_regressions(d_me,best_gn,gm,test_interactions=false)

    best_gd = extract_fx_from_formula(all_models["model_gd"][3])
    all_models["model_gdn"] = all_regressions(n_me,best_gd,gm)

    return all_models
end


function make_saver(model,perms)
    ct = coeftable(model)
    bootstrap_results = zeros(size(ct.mat,1),perms)
    function saver(i,m)
        ct = coeftable(m)
        bootstrap_results[:,i] = ct.mat[:,3]
    end
    return (bootstrap_results,saver)
end


function display_bootstrap_results(model::LinearMixedModel,
                                   zs::Matrix{Float64};
                                   niter=10000)
    println("")
    zs = sort(zs, 2)
    ct = coeftable(model)
    cis = zeros(size(ct.mat, 1), 2)
    for ro in 1:size(ct.mat, 1)
        print(ct.rownms[ro],"\t")
        print(2*length(filter(z-> z <= 0, zs[ro,:]))/niter)
        lobound = floor(0.025 * niter)
        hibound = floor(0.975 * niter)
        println("\t",zs[ro,lobound],"\t",zs[ro,hibound])
        cis[ro,1] = zs[ro,lobound]
        cis[ro,2] = zs[ro,hibound]
    end
    return (ct.rownms,cis)
end


function ztransform!(df::DataFrame, cols::Vector{Symbol})

  for c in cols
    na_ixs = Bool[isna(i) for i in df[c]]

    data::Vector{Float64} = zeros(Float64, length(df[c]))
    data[!na_ixs] = zscore(dropna(df[c]))
    df[symbol(c, "_z")] = data
  end

  df
end

ztransform(df::DataFrame, cols::Vector{Symbol}) = ztransform!(copy(df), cols)


get_follow_ups(df::DataFrame) = findin(df[:viscode],[:bl,:y1,:y2,:y3,:y4,:y5])

pcas = Symbol[ symbol("pca_", i) for i in 1:19 ]

@memoize casl() = begin
  ret = inputtable("casl_longitudinal_baseline_11Dec2015.txt")
  convert2symbols!(ret, [:subject, :viscode, :dx])
  ret[get_follow_ups(ret), :]
end

@memoize gm() = begin
  ret = inputtable("casl_longitudinal_baseline+gm_11Dec2015.txt")
  convert2symbols!(ret, [:subject, :viscode, :dx])
  normalize_variables(ret[get_follow_ups(ret), :], pcas)
end

@memoize casl_av() = begin
  ret = inputtable("casl_longitudinal_all_visits_8Dec2015.txt")
  convert2symbols!(ret, [:subject, :viscode, :dx])
  ret
end

@memoize gm_av() = begin
  ret = inputtable("casl_longitudinal_all_visits_with_GM_8Dec2015.txt")
  convert2symbols!(ret, [:subject, :viscode, :dx])
  normalize_variables(ret, pcas)
end

@memoize raw() = begin
  ret = inputtable("casl_longitudinal_raw.txt")
  convert2symbols!(ret, [:subject, :viscode, :dx])

  z_transform_infos = inputtable("z_transform_cols.txt")
  convert2symbols!(z_transform_infos, [:summed_for, :col])

  calc_order = sort(unique(z_transform_infos[:calc_order]))
  for c in calc_order
    curr_rows::Vector{Bool} = z_transform_infos[:calc_order] .== c
    cols::Vector{Symbol} = Array(z_transform_infos[curr_rows, :col])
    ztransform!(ret, cols)

    s::Symbol = z_transform_infos[curr_rows, :summed_for][1]

    z_cols::Vector{Symbol} = [symbol(c, "_z") for c in cols]
    ret[s] = sum(Matrix(ret[z_cols]), 2)[:]
  end

  ret
end

function get_best_baseline_model(df::DataFrame=gm(), baseline_only=build_all_models(casl(), gm()))

  fm::Formula = begin
    baseline_aics::Vector{Tuple{Float64, ASCIIString}} = [ (m[2][1],m[2][3]) for m in baseline_only ]
    fm_str::ASCIIString = baseline_aics[indmin(baseline_aics)][2]
    string2formula(fm_str)
  end

  fit!(lmm(fm, df))
end

function get_bootstrap_results(baseline_model::LinearMixedModel, niters::Int64=10000)
  println("*** Bootstrapping confidence intervals...")

  bootstrap_results, save_it = make_saver(baseline_model, niters)
  bootstrap(baseline_model, niters, save_it)

  bootstrap_results
end


function permute_na(da::DataArray, fn::Function=mean)
  na_ixs::Vector{Bool} = map(isna, da)
  da[na_ixs] = fn(dropna(da))
  da
end

function col_measures(df::AbstractDataFrame, col::Symbol)

  item_counts(label_fn) = begin
    cts = [label_fn(c, count(i -> i == c, df[col]))
           for c in unique(df[col])]
    join(cts, ", ")
  end

  fmt(r::Float64) = ( abs(r)>1e3 || abs(r)<1e-3 ) ? format("{:.3e}", r) : format("{:.3f}", r)

  if col == :sex
    item_counts( (c, ct) -> c == 1 ? "M: $ct" : "F: $ct")
  elseif col == :race
    item_counts( (c, ct) -> "$c: $ct")
  else
    da::DataArray = df[col]
    da = isa(da, AbstractVector{Float64}) || (convert2float(da))

    mn, std = mean_and_std(Array(permute_na(da, mean)))
    "mean: $(fmt(mn)), std: $(fmt(std))"
  end
end

demo_cols() = Symbol[symbol(n) for n in readcsv("$data_dir/input/demographics_cols.csv")]

function group_by_dx(df::DataFrame=raw())

  set_col_measures(grouped_df::AbstractDataFrame) = begin
    reduce(DataFrame(), demo_cols()) do acc, c
      acc[c] = col_measures(grouped_df, c)
      acc
    end
  end

  by(df, [:dx], set_col_measures)

end


save_group_by_dx(df::DataFrame,
                 f::AbstractString="$data_dir/step1/group_by_dx.csv") = writetable(f, df)


function calcanova2(col::Symbol, df::DataFrame=raw())
  dxdata(dx::Symbol) = Array(permute_na(df[df[:dx] .== dx, :][col]))

  calcanova(map(dxdata, [:nc, :mci, :ad])...)
end


convert_race(df::DataFrame) = [i == "B" ? 1 : 0 for i in df[:race]]


function get_different_cols(df::DataFrame=raw(), cols::Vector{Symbol}=demo_cols())
  df2::DataFrame = copy(df[:, [:dx; cols]])
  in(:race, names(df2)) && (df2[:race] = convert_race(df2))

  pval(c::Symbol) = calcanova2(c, df2).resultsInfo[1, :PValue]

  pvalsraw::Dict{Symbol, Float64} = [c => pval(c) for c in cols]
  pvalsadj::Dict{Symbol, Float64} = Dict(
    zip(keys(pvalsraw),
        padjust(collect(values(pvalsraw)), BenjaminiHochberg))
    )

  filter( (c, p) -> p < .05, pvalsadj)
end


function get_differents_df(df::DataFrame=raw(),
                           cols::Vector{Symbol}=demo_cols();
                           normalize::Bool=false)
  different_cols::Vector{Symbol} = begin
    cols_ps::Dict{Symbol, Float64} = get_different_cols(df, cols)
    [cp[1] for cp in sort(collect(cols_ps), by=cp->cp[2])]
  end

  input = copy(df[[:dx; different_cols]])
  for c in different_cols
    input[c] = permute_na(input[c], mean)
    if normalize
      input[c] = (input[c] - mean(input[c]))/std(input[c])
    end
  end
  stack(input, different_cols)
end
