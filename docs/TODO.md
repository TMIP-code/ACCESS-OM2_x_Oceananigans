- [ ] Get load balancing correct. WIP. About to try a mix between cell- and surface-loading (3D vs 2D). The issue is that 3D cell load-balancing overshot ang gave too much work to the GPU that was doing less before. Maybe some of the kernels have a workload that scales more with surface cells than with the entire interior. So overall it's probably a mix between 2D and 3D that will work best:
    - [ ] Profile vs surface balancing (2D)
    - [ ] Profile vs mix balancing (2.5D?)
- [ ] Find largest M time steop multiplier for OM2-01. I don't think there is any other way than running the NK solve for large M and crossing our fingers that it works actually...
- [ ] Implement a multi GPU NK solve. I don't think there is an Cuda-aware MPI-based linear solver so I think I need to collect the vector out of all the ranks into one big vector and call the Paridso solver on that. Maybe there is some issue with how CPUs are set up though. But we need to try this soon.
- [ ] Try to figure out why 1x8 still no scaling? This is a pain and help from NCI / ACCESS-NRI would be great.
- [ ] Combine best M + best LB + precomputed w + multiGPU for final NK solve OM2-01.
- [ ] apply adjoint tricks to get water mass fractions and ventilation fractions. This is completely orthogonal to the rest so I should be able to get it done. BTW do iI need to run the model in adjoing mode, i.e., with opposite velocities and go backwards in time for this to work? Need a bit of thought.
- [ ] Try to solve the slow restoring relaxation issue. Maybe AB3 helps? What about RK3?
- [ ] Apply xGPU NK solve OM2-01 to meltwater experiments -> paper


- [ ] Redo with OMEGA
    - [ ] 500m
    - [ ] 2500m
    - [ ] density space:
        - [ ] Use NADW, AABW etc density classes
        - [ ] Build ρ-based masks (maybe call them ρNADW, ρAABW, etc.)

- [ ] Plot basin zonal averages for eah of these OMEGA

- [ ] Plot basin zonal mean diffs for all ages x all OMEGAS;
    - [ ] In the diff look for signals of watermassses like NADW etc.
- [ ] Increase colorrange of zonal means to maybe 2000 yr.

