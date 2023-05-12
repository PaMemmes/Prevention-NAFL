import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def interpret_tree(model, train, test, df_cols):
    print('Starting interpreting...')

    test_df = pd.DataFrame(test, columns=df_cols)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_df)

    f = shap.force_plot(
        explainer.expected_value[0],
        shap_values[0],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_0.htm', f)
    plt.close()

    f = shap.force_plot(
        explainer.expected_value[1],
        shap_values[1],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_1.htm', f)
    plt.close()

    f = shap.force_plot(
        explainer.expected_value[2],
        shap_values[2],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_2.htm', f)
    plt.close()

    f = shap.force_plot(
        explainer.expected_value[3],
        shap_values[3],
        test,
        feature_names=df_cols,
        show=False)
    shap.save_html('../results/force_plot_3.htm', f)
    plt.close()

    shap.summary_plot(
        shap_values,
        test,
        plot_type="bar",
        feature_names=df_cols,
        show=False)
    f = plt.gcf()
    f.savefig('../results/summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('End interpreting...')
