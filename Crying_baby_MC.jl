### A Pluto.jl notebook ###
# v0.19.12

using Markdown
using InteractiveUtils

# ╔═╡ b4865930-441c-11ed-0bab-ade28d28791c
using Distributions, POMDPs, POMDPModelTools, QuickPOMDPs, QMDP, FIB, PointBasedValueIteration, BasicPOMCP, CSV

# ╔═╡ d789cb71-f000-452c-8c19-86a79b822497
using ParticleFilters

# ╔═╡ 342b4514-2aeb-4f6a-86af-d538d3c63de9
using DataFrames

# ╔═╡ de010dd7-0d78-40ec-9a92-50c4096e7a6b
# Uncertain inputs
begin
	N = 1000; # simulation runs
	T_steps = 20; # decision points
	
	# Rewards
	r_hungry = rand(Distributions.Uniform(-15,-5),N); # original -10
	r_feed = -5; # original -5 FIXED

	# Transition probability
	p_becomes_hungry = rand(Distributions.Uniform(0.05, 0.15),N); # original 0.1

	# Observation probabilities
	p_crying_when_hungry = rand(Distributions.Uniform(0.4, 0.9),N); # original 0.8
	p_crying_when_full = rand(Distributions.Uniform(0.05, 0.15),N); # original 0.1

	# Discount factor
	discount  = rand(Distributions.Uniform(0.89, 0.91),N); # original 0.9 

	# Output initialization
	output_qmdp = Array{Float64}(undef, N);
	output_fib = Array{Float64}(undef, N);
	output_pbvi = Array{Float64}(undef, N);
	output_pomcp = Array{Float64}(undef, N);
end

# ╔═╡ 838f0c00-f88a-4bb3-a5ea-a0423ff8e2df
begin
	@enum State hungry full
	@enum Action feed ignore
	@enum Observation crying quiet
end

# ╔═╡ cc67621d-abc1-46cb-8978-e6d4fe90ae06
import POMDPTools: HistoryRecorder

# ╔═╡ 9d6aa874-897c-4e8e-a01d-3a342de3b36a
# Monte Carlo
for i in 1:N

	# POMDP
    
			pomdp = QuickPOMDP(
		    states       = [hungry, full],  # 𝒮
		    actions      = [feed, ignore],  # 𝒜
		    observations = [crying, quiet], # 𝒪
		    initialstate = [full],          # Deterministic initial state
		    
		
		    transition = function T(s, a)
		        if a == feed
		            return SparseCat([hungry, full], [0, 1])
		        elseif s == hungry && a == ignore
		            return SparseCat([hungry, full], [1, 0])
		        elseif s == full && a == ignore
		            return SparseCat([hungry, full], [p_becomes_hungry[i], 1 - p_becomes_hungry[i]])
		        end
		    end,
		
		    observation = function O(s, a, s′)
		        if s′ == hungry
		            return SparseCat([crying, quiet], [p_crying_when_hungry[i], 1 - p_crying_when_hungry[i]])
		        elseif s′ == full
		            return SparseCat([crying, quiet], [p_crying_when_full[i], 1 - p_crying_when_full[i]])
		        end
		    end,
		
		    reward = (s,a)->(s == hungry ? r_hungry[i] : 0) + (a == feed ? r_feed : 0)
		)

	
	# solve POMDP

		# QMDP
	
		qmdp_solver = QMDPSolver();
		qmdp_policy = solve(qmdp_solver, pomdp)
		
		recorder = HistoryRecorder(max_steps=T_steps);
		updater = BootstrapFilter(pomdp, 1000);
	
		history_qmdp = simulate(recorder,
		               pomdp,
		               qmdp_policy,
		               updater,
					   initialstate(pomdp),
	                   rand(initialstate(pomdp)))
		# FIB
		fib_solver = FIBSolver()
		fib_policy = solve(fib_solver, pomdp)
		history_fib = simulate(recorder,
		               pomdp,
		               fib_policy,
		               updater,
					   initialstate(pomdp),
	                   rand(initialstate(pomdp)))
		#PBVI
#		pbvi_solver = PBVISolver()
#		pbvi_policy = solve(pbvi_solver, pomdp)
#		history_fib = simulate(recorder,
#		               pomdp,
#		               pbvi_policy,
#		               updater,
#					   initialstate(pomdp),
#	                   rand(initialstate(pomdp)))

		#POMCP
		pomcp_solver = POMCPSolver()
		pomcp_planner = solve(pomcp_solver, pomdp);
		history_pomcp = simulate(recorder,
		               pomdp,
		               pomcp_planner,
		               updater,
					   initialstate(pomdp),
	                   rand(initialstate(pomdp)))
	
	# disounting reward

	discount_factor = Array{Float64}(undef, T_steps);
	for k in 1:T_steps
		discount_factor[k] = discount[i] ^ k;
	end
	
	output_qmdp[i] = sum(history_qmdp[:r] .* discount_factor);
	output_fib[i] = sum(history_fib[:r] .* discount_factor);
#	output_pbvi[i] = sum(history_pbvi[:r] .* discount_factor);
	output_pomcp[i] = sum(history_pomcp[:r] .* discount_factor);
	
end

# ╔═╡ 9190ac90-5385-4487-af89-24cd6bcd0f54
	df_qmdp = DataFrame([output_qmdp r_hungry p_becomes_hungry p_crying_when_hungry p_crying_when_full discount], :auto)

# ╔═╡ 7b77711a-0d68-46d1-be80-063a58013f97
	df_fib = DataFrame([output_fib r_hungry p_becomes_hungry p_crying_when_hungry p_crying_when_full discount], :auto)

# ╔═╡ eb514bf7-5ee0-41d7-b731-7a7f0c86a873
	df_pbvi = DataFrame([output_pbvi r_hungry p_becomes_hungry p_crying_when_hungry p_crying_when_full discount], :auto)

# ╔═╡ 90c025a0-468b-4bd0-8c1d-82e3c24967da
	df_pomcp = DataFrame([output_pomcp r_hungry p_becomes_hungry p_crying_when_hungry p_crying_when_full discount], :auto)

# ╔═╡ 26323774-5cce-418c-84fa-65afe437ab36
begin
	CSV.write("Desktop\\qmdp.csv", df_qmdp)
	CSV.write("Desktop\\fib.csv", df_fib)
#	CSV.write("Desktop\\pbvi.csv", df_pbvi)
	CSV.write("Desktop\\pomcp.csv", df_pomcp)
end
