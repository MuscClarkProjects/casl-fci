using Base.Test
using DataFrames

include("../src/casl_lmm_fit.jl")

df = DataFrame(a=@data([1, 2, 3, NA]))

df_z = ztransform(df, [:a])

@test df_z[:a_z] == [-1, 0, 1, 0]
