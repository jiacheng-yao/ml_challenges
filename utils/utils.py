from typing import Optional
import logging
import logging.config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pandas import DataFrame
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
sns.set_style('whitegrid')

def config_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    filter_snowflake_logs: Optional[bool] = False,
    filter_helpers_datamodel_logs: Optional[bool] = False,
):
    # helper function to standardize logging outputs
    filter_pydro = logging.Filter()
    filter_pydro.__setattr__(
        "filter",
        lambda record: (
            (record.name != "pydro.backends.sqldb")
            | ("Result ready for fetching" not in record.msg)
        ),
    )

    handlers = []
    handler_stdout = logging.StreamHandler()
    handler_stdout.setLevel(level)
    handler_stdout.addFilter(filter_pydro)
    snowflake_filter = logging.Filter()
    if filter_snowflake_logs:
        snowflake_filter.__setattr__(
            "filter", lambda record: ("snowflake" not in record.name)
        )
        handler_stdout.addFilter(snowflake_filter)
    helpers_datamodel_filter = logging.Filter()
    if filter_helpers_datamodel_logs:
        helpers_datamodel_filter.__setattr__(
            "filter", lambda record: ("helpers_datamodel" not in record.name)
        )
        handler_stdout.addFilter(helpers_datamodel_filter)
    handlers.append(handler_stdout)

    if log_file:
        handler_file = logging.FileHandler(log_file)
        handler_file.addFilter(filter_pydro)
        if filter_snowflake_logs:
            handler_file.addFilter(snowflake_filter)
        if filter_helpers_datamodel_logs:
            handler_file.addFilter(helpers_datamodel_filter)
        handler_file.setLevel(level)
        handlers.append(handler_file)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def missing_value_checker(df):
    # Check if there are missing values in the columns:
    for col in df.columns:
        tmp_flag = 'Numerical'
        if df[col].dtype == object:
            tmp_flag = 'Categorical'
        logging.info('{}: {} ({})'.format(col, str(df[col].isnull().sum()/float(df.shape[0])), tmp_flag))


def unique_value_printer(df):
    # See how many unique values there are for each columns,
    # and if there is only one unique value, we drop the column
    cols_to_drop = []
    for col in df.columns:
        tmp_num_unique = len(df[col].unique())
        tmp_flag = 'Numerical'
        if df[col].dtype == object:
            tmp_flag = 'Categorical'
        logging.info('{}: {} ({})'.format(col, str(tmp_num_unique), tmp_flag))
        if (tmp_num_unique==1):
            cols_to_drop.append(col)



def plot_general_dist(
    df,
    col,
    title=None,
    outfile=None,
    figsize_x=15,
    figsize_y=7
    ):
    # plot general histogram
    fig = plt.figure(figsize=(figsize_x,figsize_y))
    ax = fig.add_subplot(111)
    if title is not None:
        sns.distplot(df[col]).set_title(title)
    if outfile is not None:
        plt.savefig(outfile)

def plot_general_line(df, col_x, col_y, title=None, outfile=None):
    # plot general histogram
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    if title is not None:
        sns.lineplot(x=col_x, y=col_y, data=df).set_title(title)
    if outfile is not None:
        plt.savefig(outfile)

def plot_comparison_dist(l_dfs, col, l_labels, title=None, outfile=None):
    # plot comparison histogram
    # len(l_dfs) and len(l_labels) need to be the same
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    for i in range(len(l_dfs)):
        sns.distplot(l_dfs[i][col], hist=False, rug=False, label=l_labels[i]).set_title(title)

    if outfile is not None:
        plt.savefig(outfile)

def plot_general_bar(
    df,
    group_col,
    agg_col,
    title=None,
    outfile=None,
    rotation=0,
    xlabel="",
    ylabel="",
    figsize_x=15,
    figsize_y=7,
    to_sort=False,
    ascending=False,
    n_limit=10
    ):
    # plot general barplot
    df_grouped = df.groupby([group_col]).count().reset_index()

    if to_sort:
        df_grouped = df_grouped.sort_values(by=agg_col, ascending=ascending).head(n_limit)

    fig, axis = plt.subplots(1,1,figsize=(figsize_x, figsize_y))

    g = sns.barplot(x=group_col, y=agg_col, data=df_grouped, order=df_grouped[group_col].unique(), ax=axis)
    axis.set_xticks(range(len(df_grouped[group_col].unique())))
    axis.set_xticklabels(df_grouped[group_col].unique(), rotation=rotation)

    g.set_xlabel(xlabel)
    g.set_ylabel(ylabel)

    if title is not None:
        axis.set_title(title)
    if outfile is not None:
        plt.savefig(outfile)

def plot_comparison_bar(df, group_col, agg_col, hue_col, title=None, outfile=None):
    df_grouped_by_col = df.groupby([group_col, hue_col]).agg({agg_col:'count'}).reset_index()

    g = sns.catplot(x=group_col, y=agg_col, hue=hue_col, data=df_grouped_by_col,
                    kind="bar", palette="muted", height=6, aspect=2)
    g.despine(left=True)
    g.set_ylabels("Count")

    if title is not None:
        g.fig.suptitle(title)
    if outfile is not None:
        g.fig.savefig(outfile)

def plot_feature_importance(importance, names, model_type, outfile=None):
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(18,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ': Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')

    if outfile is not None:
        plt.savefig(outfile)

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y', outfile=None):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sns.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim

    if outfile is not None:
        plt.savefig(outfile)
    plt.show()
