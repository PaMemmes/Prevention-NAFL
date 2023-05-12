import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_cats_plots(df):
    plt.figure(figsize=(21.2, 10))
    plt.subplot(2, 4, 1)
    sns.countplot(x=df['Stage'], hue=df['Sex'], palette='Blues', alpha=0.9)

    plt.subplot(2, 4, 2)
    sns.countplot(x=df['Stage'], hue=df['Drug'], palette='Blues', alpha=0.9)

    plt.subplot(2, 4, 3)
    sns.countplot(x=df['Stage'], hue=df['Ascites'], palette='Blues', alpha=0.9)

    plt.subplot(2, 4, 4)
    sns.countplot(
        x=df['Stage'],
        hue=df['Hepatomegaly'],
        palette='Blues',
        alpha=0.9)

    plt.subplot(2, 4, 5)
    sns.countplot(x=df['Stage'], hue=df['Spiders'], palette='Blues', alpha=0.9)

    plt.subplot(2, 4, 6)
    sns.countplot(x=df['Stage'], hue=df['Edema'], palette='Blues', alpha=0.9)

    plt.subplot(2, 4, 7)
    sns.countplot(x=df['Stage'], hue=df['Status'], palette='Blues', alpha=0.9)
    plt.savefig("../results/categoricals.png", bbox_inches="tight")
    plt.close()


def show_dists(df):
    plt.figure(figsize=(20.6, 15))

    plt.subplot(3, 3, 1)
    sns.kdeplot(x=df['Bilirubin'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 2)
    sns.kdeplot(x=df['Cholesterol'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 3)
    sns.kdeplot(x=df['Albumin'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 4)
    sns.kdeplot(x=df['Copper'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 5)
    sns.kdeplot(x=df['Alk_Phos'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 6)
    sns.kdeplot(x=df['SGOT'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 7)
    sns.kdeplot(x=df['Tryglicerides'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 8)
    sns.kdeplot(x=df['Platelets'], hue=df['Stage'], fill=True)

    plt.subplot(3, 3, 9)
    sns.kdeplot(x=df['Prothrombin'], hue=df['Stage'], fill=True)
    plt.savefig("../results/categoricals.png", bbox_inches="tight")
    plt.close()


if __name__ == '__main__':

    df = pd.read_csv('../data/kaggle_cirrhosis.csv')
    ax = sns.histplot(x=df['Stage'], discrete=True, binwidth=0.4)
    print(df)
    ax.figure.savefig("../results/stage.png", bbox_inches="tight")
    plt.close()

    corr = df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                     square=True, linewidths=.5, cbar_kws={"shrink": .5})
    ax.figure.savefig("../results/corr.png", bbox_inches="tight")
    plt.close()

    make_cats_plots(df)
    show_dists(df)
