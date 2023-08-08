import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import torch

def interpret_tree(model, train, test, df_cols, nn=False) -> None:
    print('Starting interpreting...')
    
    test_df = test
    inf_inds = torch.randperm(len(test))[:40]
    inf_data = train[inf_inds]

    if nn is True:
        explainer = shap.DeepExplainer(model, data=inf_data)
        model_name = 'ffn'
        test = test.numpy() # Can't use html plot with tensors
    else:
        explainer = shap.TreeExplainer(model, data=inf_data)
        model_name = 'tree'
    shap_values = explainer.shap_values(test_df)

    df_shap = pd.DataFrame(shap_values[0], columns=df_cols)
    df_shap.to_excel('../results/shaps_0' + model_name + '.xlsx', index=False)
    df_shap = pd.DataFrame(shap_values[1], columns=df_cols)
    df_shap.to_excel('../results/shaps_1' + model_name + '.xlsx', index=False)
    df_shap = pd.DataFrame(shap_values[2], columns=df_cols)
    df_shap.to_excel('../results/shaps_2' + model_name + '.xlsx', index=False)

    f = shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_0_' + model_name + '.htm', f)
    plt.close()

    f = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_1_' + model_name + '.htm', f)
    plt.close()

    f = shap.force_plot(
        explainer.expected_value[2],
        shap_values[2],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_2_' + model_name + '.htm', f)
    plt.close()

    shap.summary_plot(
        shap_values,
        test,
        plot_type="bar",
        feature_names=df_cols,
        show=False)
    f = plt.gcf()
    f.savefig('../results/summary_bar_' + model_name + '.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('End interpreting...')
