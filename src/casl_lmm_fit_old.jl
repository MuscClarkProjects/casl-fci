using DataFrames, MixedModels, MultivariateStats # , PyPlot

srand(42)

function zed(v::DataArray{Float64,1})
    return (v - mean(v))/std(v)
end

function mlr(dv,iv)
    iv = hcat(ones(length(dv),1),iv)
    (n,p) = size(iv)
    term1 = inv(iv' * iv)
    beta = term1 * iv' * dv
    yhat = iv * beta
    e = dv - yhat
    rss = e' * e
    s_sq = rss/(n-p)
    disp = term1 * s_sq[1]
    y_bar = mean(dv)
    sst = sum((dv - y_bar).^2)
    mse = sum((yhat - y_bar).^2)
    rsq = mse/sst
    f = (mse/(p-1)) ./ (rss/(n-p))
    ts = beta ./ sqrt(diag(disp))
    return Dict{Any,Any}( :beta => beta, :yhat => yhat, :residuals => e, :tstats => ts, :fstat => f[1], :Rsq => rsq )
end

function convert2symbols(df,sym)
    confun = x -> isna(x) ? NA : convert(Symbol,x)
    return [ confun(df[i,sym]) for i in 1:size(df,1) ]
end

function string2formula(s)
    return eval(parse(s))
end

# function combine_main_effects(master_me)
    # println(master_me)
#    all_me = []
#    for i in 0:length(master_me)
        # println(i)
#        combos = [ c for c in master_me ]
#        append!(all_me,combos)
#    end
#    return all_me
# end

function add_fci(col)
    if !contains(col,"fci")
        push!(col,"fci")
    end
    return col
end

function combine_main_effects(master_me;add_var=true)
    if add_var
        return [ unique(push!(c,"fci")) for c in master_me ]
    else
        return [ c for c in master_me ]
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

function convert_bool_to_01(b)
    if b
        return 1
    else
        return 0
    end
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

function normalize_variables(df,varnames)
    for varname in varnames
        (mu,sd) = mean_and_std(dropna(df[varname]))
        if sd != 0
            df[varname] = (df[varname]-mu)/sd
        end
    end
    return df
end

pcas = [ convert(Symbol,string("pca_",i)) for i in 1:19 ]

casl = readtable("../data/input/casl_longitudinal_baseline_11Dec2015.txt",separator='\t')
casl[:subject] = convert2symbols(casl,:subject)
casl[:viscode] = convert2symbols(casl,:viscode)
fus = findin(casl[:viscode],[:bl,:y1,:y2,:y3,:y4,:y5])
casl = casl[fus,:]
# older = findin(casl[:age],57:100)
# casl = casl[older,:]
casl[:dx] = convert2symbols(casl,:dx)

gm = readtable("../data/input/casl_longitudinal_baseline+gm_11Dec2015.txt",separator = '\t')
gm[:subject] = convert2symbols(gm,:subject)
gm[:viscode] = convert2symbols(gm,:viscode)
fus = findin(gm[:viscode],[:bl,:y1,:y2,:y3,:y4,:y5])
gm = gm[fus,:]
gm[:dx] = convert2symbols(gm,:dx)
gm = normalize_variables(gm,pcas)

casl_av = readtable("../data/input/casl_longitudinal_all_visits_8Dec2015.txt",separator='\t')
casl_av[:subject] = convert2symbols(casl_av,:subject)
casl_av[:viscode] = convert2symbols(casl_av,:viscode)
# fus = findin(casl_av[:viscode],[:bl,:y1,:y2,:y3,:y4,:y5])
# casl_av = casl_av[fus,:]
casl_av[:dx] = convert2symbols(casl_av,:dx)

gm_av = readtable("../data/input/casl_longitudinal_all_visits_with_GM_8Dec2015.txt",separator = '\t')
gm_av[:subject] = convert2symbols(gm_av,:subject)
gm_av[:viscode] = convert2symbols(gm_av,:viscode)
# fus = findin(gm_av[:viscode],[:bl,:y1,:y2,:y3,:y4,:y5])
# gm_av = gm_av[fus,:]
gm_av[:dx] = convert2symbols(gm_av,:dx)
gm_av = normalize_variables(gm_av,pcas)

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

baseline_only = build_all_models(casl,gm)
# all_visits = build_all_models(casl_av,gm_av,n_me=["atn", "exe", "mem", "sem", "time_elapsed", "fci_bl"])

println("*** Bootstrapping confidence intervals...")

function permute_significance(formula,df;niter=10000)
    model = fit!(lmm(string2formula(formula),df))
    ct = coeftable(model)
    zs = zeros(size(ct.mat,1),niter)
    zs[:,1] = ct.mat[:,3]
    for it in 2:niter
        roze = sample(1:size(df,1),size(df,1),replace=false)
        nudf = df[roze,:]
        m = fit!(lmm(string2formula(formula),nudf))
        if it % 1000 == 0
            println(" $it ")
        end
        ct = coeftable(m)
        zs[:,it] = ct.mat[:,3]
    end

    zs = sort(zs,2)
    ct = coeftable(model)
    for ro in 1:size(ct.mat,1)
        print(ct.rownms[ro],"\t")
        print(length(filter(z-> z <= 0, zs[ro,:]))/niter)
        lobound = floor(0.025 * niter)
        hibound = floor(0.975 * niter)
        println("\t",zs[ro,lobound],"\t",zs[ro,hibound])
    end
end

function bootstrap_CI(formula,df;niter=10000)
    model = fit!(lmm(string2formula(formula),df))
    ct = coeftable(model)
    zs = zeros(size(ct.mat,1),niter)
    zs[:,1] = ct.mat[:,3]
    for it in 2:niter
        roze = sample(1:size(df,1),size(df,1))
        nudf = df[roze,:]
        m = fit!(lmm(string2formula(formula),nudf))
        if it % 1000 == 0
            print(" .")
        end
        ct = coeftable(m)
        zs[:,it] = ct.mat[:,3]
    end
    println("")
    zs = sort(zs,2)
    ct = coeftable(model)
    for ro in 1:size(ct.mat,1)
        print(ct.rownms[ro],"\t")
        print(2*length(filter(z-> z <= 0, zs[ro,:]))/niter)
        lobound = floor(0.025 * niter)
        hibound = floor(0.975 * niter)
        println("\t",zs[ro,lobound],"\t",zs[ro,hibound])
    end
end

function bootstrap_lmm_CI(formula,df;niter=10000,verbose=false)
    model = fit!(lmm(string2formula(formula),df))
    ct = coeftable(model)
    zs = zeros(size(ct.mat,1),niter)
    zs[:,1] = ct.mat[:,3]
    subs = unique(df[:subject])
    for it in 2:niter
        subsample = sample(subs,length(subs))
        roze = findin(df[:subject],subsample)
        nudf = df[roze,:]
        m = fit!(lmm(string2formula(formula),nudf))
        if verbose
            println(m)
        end
        if it % 1000 == 0
            print(" .")
        end
        ct = coeftable(m)
        zs[:,it] = ct.mat[:,3]
    end
    println("")
    zs = sort(zs,2)
    ct = coeftable(model)
    cis = zeros(size(ct.mat,1),2)
    for ro in 1:size(ct.mat,1)
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

function make_saver(model,perms)
    ct = coeftable(model)
    bootstrap_results = zeros(size(ct.mat,1),perms)
    function saver(i,m)
        ct = coeftable(m)
        bootstrap_results[:,i] = ct.mat[:,3]
    end
    return (bootstrap_results,saver)
end

function display_bootstrap_results(model,zs;niter=10000)
    println("")
    zs = sort(zs,2)
    ct = coeftable(model)
    cis = zeros(size(ct.mat,1),2)
    for ro in 1:size(ct.mat,1)
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

baseline_aics = [ (m[2][1],m[2][3]) for m in baseline_only ]
# all_visits_aics = [ (m[2][1],m[2][3]) for m in all_visits ]

best_baseline_model = baseline_aics[indmin(baseline_aics)][2]
# best_all_visits_model = all_visits_aics[indmin(all_visits_aics)][2]

bl = fit!(lmm(string2formula(best_baseline_model),gm))
# av = fit!(lmm(string2formula(best_all_visits_model),gm_av));
# ct_av = coeftable(av);

# println(ct_av)

(bootstrap_results,save_it) = make_saver(bl,10000)
bootstrap(bl,10000,save_it)
display_bootstrap_results(bl,bootstrap_results)

# (bootstrap_results_av,saver_av) = make_saver(av,10000)
# bootstrap(av,10000,saver_av)
# display_bootstrap_results(av,bootstrap_results_av)

# sigs_baseline = bootstrap_lmm_CI(best_baseline_model,gm)
# srand(123)
# sigs_all_visits = bootstrap_lmm_CI(best_all_visits_model,gm_av)

