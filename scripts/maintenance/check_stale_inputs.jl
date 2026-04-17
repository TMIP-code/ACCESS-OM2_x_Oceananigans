#!/usr/bin/env julia
# Check which preprocessed velocity inputs are stale after the grid y-dimension fix.
#
# Stale files have v's y-dimension == u's y-dimension (both Center-like).
# Up-to-date files have v's y-dimension == u's y-dimension + 1 (v on Face).
#
# Usage:
#   julia scripts/maintenance/check_stale_inputs.jl                   # scan all
#   julia scripts/maintenance/check_stale_inputs.jl --archive-stale   # move stale dirs to archive/

using JLD2

archive_stale = "--archive-stale" in ARGS

base_dir = joinpath(@__DIR__, "..", "..", "preprocessed_inputs")
archive_dir = joinpath(@__DIR__, "..", "..", "archive", "preprocessed_inputs")

stale_dirs = String[]
uptodate_dirs = String[]
error_dirs = String[]

for pm_dir in readdir(base_dir; join = true)
    isdir(pm_dir) || continue
    pm = basename(pm_dir)
    for exp_dir in readdir(pm_dir; join = true)
        isdir(exp_dir) || continue
        exp = basename(exp_dir)

        for tw_dir in readdir(exp_dir; join = true)
            isdir(tw_dir) || continue
            tw = basename(tw_dir)
            tw in ("partitions", "plots") && continue

            yearly_dir = joinpath(tw_dir, "yearly")
            isdir(yearly_dir) || continue

            u_file = joinpath(yearly_dir, "u_from_mass_transport_yearly.jld2")
            v_file = joinpath(yearly_dir, "v_from_mass_transport_yearly.jld2")
            isfile(u_file) && isfile(v_file) || continue

            try
                u_shape = jldopen(f -> size(f["u"]), u_file, "r")
                v_shape = jldopen(f -> size(f["v"]), v_file, "r")

                label = "$pm/$exp/$tw"
                if v_shape[2] == u_shape[2]
                    println("STALE   $label  u=$(u_shape) v=$(v_shape)")
                    push!(stale_dirs, tw_dir)
                elseif v_shape[2] == u_shape[2] + 1
                    println("OK      $label  u=$(u_shape) v=$(v_shape)")
                    push!(uptodate_dirs, tw_dir)
                else
                    println("???     $label  u=$(u_shape) v=$(v_shape)  (unexpected difference)")
                    push!(error_dirs, tw_dir)
                end
            catch e
                println("ERROR   $pm/$exp/$tw  $e")
                push!(error_dirs, tw_dir)
            end
        end

        # Check partitions too
        part_base = joinpath(exp_dir, "partitions")
        isdir(part_base) || continue
        for part_dir in readdir(part_base; join = true)
            isdir(part_dir) || continue
            part = basename(part_dir)
            u_rank0 = joinpath(part_dir, "u_from_mass_transport_monthly_rank0.jld2")
            v_rank0 = joinpath(part_dir, "v_from_mass_transport_monthly_rank0.jld2")
            isfile(u_rank0) && isfile(v_rank0) || continue

            try
                u_shape = jldopen(f -> size(f["data/1"]), u_rank0, "r")
                v_shape = jldopen(f -> size(f["data/1"]), v_rank0, "r")

                label = "$pm/$exp/partitions/$part"
                if v_shape[2] == u_shape[2]
                    println("STALE   $label  u=$(u_shape) v=$(v_shape)")
                    push!(stale_dirs, part_dir)
                elseif v_shape[2] == u_shape[2] + 1
                    println("OK      $label  u=$(u_shape) v=$(v_shape)")
                    push!(uptodate_dirs, part_dir)
                else
                    println("???     $label  u=$(u_shape) v=$(v_shape)  (unexpected difference)")
                    push!(error_dirs, part_dir)
                end
            catch e
                println("ERROR   $pm/$exp/partitions/$part  $e")
                push!(error_dirs, part_dir)
            end
        end

        # Check per-TW partitions
        for tw_dir in readdir(exp_dir; join = true)
            isdir(tw_dir) || continue
            tw = basename(tw_dir)
            tw in ("partitions", "plots") && continue
            tw_part_base = joinpath(tw_dir, "partitions")
            isdir(tw_part_base) || continue

            for part_dir in readdir(tw_part_base; join = true)
                isdir(part_dir) || continue
                part = basename(part_dir)
                u_rank0 = joinpath(part_dir, "u_from_mass_transport_monthly_rank0.jld2")
                v_rank0 = joinpath(part_dir, "v_from_mass_transport_monthly_rank0.jld2")
                isfile(u_rank0) && isfile(v_rank0) || continue

                try
                    u_shape = jldopen(f -> size(f["data/1"]), u_rank0, "r")
                    v_shape = jldopen(f -> size(f["data/1"]), v_rank0, "r")

                    label = "$pm/$exp/$tw/partitions/$part"
                    if v_shape[2] == u_shape[2]
                        println("STALE   $label  u=$(u_shape) v=$(v_shape)")
                        push!(stale_dirs, part_dir)
                    elseif v_shape[2] == u_shape[2] + 1
                        println("OK      $label  u=$(u_shape) v=$(v_shape)")
                        push!(uptodate_dirs, part_dir)
                    else
                        println("???     $label  u=$(u_shape) v=$(v_shape)  (unexpected difference)")
                        push!(error_dirs, part_dir)
                    end
                catch e
                    println("ERROR   $pm/$exp/$tw/partitions/$part  $e")
                    push!(error_dirs, part_dir)
                end
            end
        end
    end
end

println("\n=== Summary ===")
println("Up-to-date: $(length(uptodate_dirs))")
println("Stale:      $(length(stale_dirs))")
println("Errors:     $(length(error_dirs))")

if archive_stale && !isempty(stale_dirs)
    println("\nArchiving stale directories:")
    for d in stale_dirs
        rel = relpath(d, base_dir)
        dest = joinpath(archive_dir, rel)
        mkpath(dirname(dest))
        println("  mv $d → $dest")
        mv(d, dest; force = true)
    end
    println("Done — $(length(stale_dirs)) directories archived to archive/preprocessed_inputs/.")
elseif !isempty(stale_dirs)
    println("\nRe-run with --archive-stale to move stale directories to archive/preprocessed_inputs/.")
end
