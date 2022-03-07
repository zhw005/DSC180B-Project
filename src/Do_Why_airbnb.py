import dowhy

def do_why(data, treatment, outcome, common_causes):
    # 1. Model
    model= dowhy.CausalModel(
            data = data,
            treatment= treatment,
            outcome= outcome,
            common_causes = common_causes)

    # 2. Identify
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

    # 3. Estimate
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression",target_units="ate")
    # ATE = Average Treatment Effect
    # ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
    # ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
    print(estimate)

    # 4. Refute

    # Radom Common Cause:- Adds randomly drawn covariates to data and re-runs the analysis to see if the causal estimate changes or not.
    # If our assumption was originally correct then the causal estimate shouldn’t change by much.
    refute1_results=model.refute_estimate(identified_estimand, estimate,
            method_name="random_common_cause")
    print(refute1_results)

    # Placebo Treatment Refuter:- Randomly assigns any covariate as a treatment and re-runs the analysis.
    # If our assumptions were correct then this newly found out estimate should go to 0.
    #refute2_results=model.refute_estimate(identified_estimand, estimate,
    #                                      method_name="placebo_treatment_refuter")
    #print(refute2_results)

    # Data Subset Refuter:- Creates subsets of the data(similar to cross-validation) and checks whether the causal estimates vary across subsets.
    # If our assumptions were correct there shouldn’t be much variation.
    refute3_results=model.refute_estimate(identified_estimand, estimate,
            method_name="data_subset_refuter")
    print(refute3_results)
