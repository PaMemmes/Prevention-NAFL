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

    #f = shap.force_plot(explainer.expected_value, shap_values, test, feature_names=df_cols, show=False)
    #shap.save_html('../results/force_plot.htm', f)
    #plt.close()
    #print('After force plot')
    shap.summary_plot(shap_values, test, plot_type="bar", feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig('../results/summary_bar.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('After first summary plot')
    shap.summary_plot(shap_values, test, feature_names=df_cols, show=False)
    f = plt.gcf()
    f.savefig('../results/summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('End interpreting...')