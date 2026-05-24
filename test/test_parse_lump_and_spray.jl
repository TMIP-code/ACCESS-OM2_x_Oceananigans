"""
Unit tests for `parse_lump_and_spray()` in src/shared_utils/config.jl.

Run on a login node (no MPI/CUDA needed):
    julia --project test/test_parse_lump_and_spray.jl
"""

using TOML  # config.jl uses TOML at top of file
include("../src/shared_utils/config.jl")

function expect_off(s)
    r = parse_lump_and_spray(s)
    @assert r.on == false       "expected on=false for $(repr(s)), got $(r.on)"
    @assert r.di == 0           "expected di=0 for $(repr(s)), got $(r.di)"
    @assert r.dj == 0           "expected dj=0 for $(repr(s)), got $(r.dj)"
    @assert r.dk == 0           "expected dk=0 for $(repr(s)), got $(r.dk)"
    @assert r.tag == "prec"     "expected tag=\"prec\" for $(repr(s)), got $(repr(r.tag))"
    return @assert r.dir_suffix == ""  "expected dir_suffix=\"\" for $(repr(s)), got $(repr(r.dir_suffix))"
end

function expect_on(s, di_exp, dj_exp)
    r = parse_lump_and_spray(s)
    @assert r.on == true                            "expected on=true for $(repr(s)), got $(r.on)"
    @assert r.di == di_exp                          "expected di=$di_exp for $(repr(s)), got $(r.di)"
    @assert r.dj == dj_exp                          "expected dj=$dj_exp for $(repr(s)), got $(r.dj)"
    @assert r.dk == 1                               "expected dk=1 for $(repr(s)), got $(r.dk)"
    @assert r.tag == "Q$(di_exp)x$(dj_exp)"         "expected tag=Q$(di_exp)x$(dj_exp) for $(repr(s)), got $(repr(r.tag))"
    return @assert r.dir_suffix == "_Q$(di_exp)x$(dj_exp)" "expected dir_suffix=_Q$(di_exp)x$(dj_exp) for $(repr(s)), got $(repr(r.dir_suffix))"
end

function expect_error(s, needle)
    return try
        parse_lump_and_spray(s)
        error("expected parse_lump_and_spray($(repr(s))) to error, but it returned successfully")
    catch e
        msg = sprint(showerror, e)
        occursin(needle, msg) ||
            error("expected error for $(repr(s)) to contain $(repr(needle)); got: $msg")
    end
end

@info "parse_lump_and_spray: off cases"
expect_off("no")
expect_off("NO")
expect_off("No")

@info "parse_lump_and_spray: on cases"
expect_on("2x2", 2, 2)
expect_on("5x5", 5, 5)
expect_on("4x4", 4, 4)
expect_on("3x3", 3, 3)
expect_on("6x6", 6, 6)
expect_on("10x10", 10, 10)
expect_on("5X5", 5, 5)            # case-insensitive
expect_on("3x7", 3, 7)            # asymmetric factors

@info "parse_lump_and_spray: error cases"
expect_error("yes", "no longer supported")
expect_error("YES", "no longer supported")
expect_error("", "must be 'no' or")
expect_error("5", "must be 'no' or")
expect_error("5x", "must be 'no' or")
expect_error("x5", "must be 'no' or")
expect_error("5xfoo", "must be 'no' or")
expect_error("0x5", "positive integers")
expect_error("5x0", "positive integers")

@info "All parse_lump_and_spray tests passed"
