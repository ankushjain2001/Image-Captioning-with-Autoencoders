import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import altair as alt

# Loss plot
def plot_loss(source, x_col, y_col, cat_col, y_dom):
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=[x_col], empty='none')
    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x=x_col+':Q',
        y=alt.Y(y_col+':Q', scale=alt.Scale(domain=y_dom)),
        color=cat_col+':N'
    )
    #line.encode(alt.Y(y_col, scale=alt.Scale(domain=[0.5, 0.6])))
    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x=x_col+':Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, y_col+':Q', alt.value(' '))
    )
    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x=x_col+':Q',
    ).transform_filter(
        nearest
    )
    # Put the five layers into a chart and bind the data
    return alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=800, height=300
    )
	

# tSNE Plot
def plot_tsne(source, x_col, y_col, category, perplexities, img_name):

    cats = list(source[category].unique())

    # Base
    base = alt.Chart(source).mark_point(filled=True).encode(
        x=x_col+':Q',
        y=y_col+':Q',
        tooltip=img_name+':N'
    )


    # A slider filter
    perp_slider = alt.binding_range(min=10, max=40, step=10)
    slider_selection = alt.selection_single(bind=perp_slider, fields=[perplexities], name="Change")

    # Color changing marks
    rating_radio = alt.binding_radio(options=cats)
    rating_select = alt.selection_single(fields=[category], bind=rating_radio, name="Filter")
    rating_color_condition = alt.condition(rating_select,
                          alt.Color(category+':N', legend=None),
                          alt.value('lightgray'))

    highlight_ratings = base.add_selection(
        rating_select, slider_selection
    ).encode(
        color=rating_color_condition
    ).transform_filter(
        slider_selection
    ).properties(title="tSNE Scatter Plot")

    return highlight_ratings.properties(width=800, height=300)


def plottingFeatureMaps(img_idx, model):
    test_img = X_train_PP[img_idx]
    test_img = np.expand_dims(test_img, axis=0)
    feature_maps = model.predict(test_img)
    map_cnt = 0
    square = 8
    for fmap in feature_maps:
        num_fmap = (fmap.shape[3])
        ix = 1
        fig = plt.figure(figsize=(10,10))
        fig.suptitle('Visualizing ' + encoder_layer_names[map_cnt], fontsize=20)
        #print('Visualizing ' + encoder_layer_names[map_cnt])
        map_cnt+=1
        for _ in range(int(num_fmap/8)):
            for _ in range(8):
                ax = plt.subplot(square,square,ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(fmap[0,:,:,ix-1])
                ix+=1
        plt.show()



def plottingReconstructionMap(img_idx):
    test_img = X_train_PP[img_idx]
    test_img = cv2.normalize(test_img, test_img,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    reco_img = X_train_reconst[img_idx]
    reco_img = cv2.normalize(reco_img, reco_img,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    reco_img = cv2.cvtColor(reco_img, cv2.COLOR_BGR2RGB)
    
    fig = plt.figure(figsize=(8, 4))
    fig.suptitle("Reconstructed Image", fontsize=20)
    ax = plt.subplot(1, 2, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(test_img)
    ax = plt.subplot(1, 2, 2)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.imshow(reco_img)
    plt.show()


# Function to plot the recommended images
def plot_nearest_images(img_query, imgs_retrieval, outFile, cap, tags, id_x):
    n_retrieval = len(imgs_retrieval)+1
    fig = plt.figure(figsize=(2.5*n_retrieval, 3))
    fig.suptitle("Test Results "+str(id_x), fontsize=20)
    # Plot query image
    ax = plt.subplot(1, n_retrieval, 1)
    img_query = cv2.resize(img_query, image_dims[:2])
    img_query = cv2.cvtColor(img_query, cv2.COLOR_BGR2RGB)
    plt.imshow(img_query)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
        ax.spines[axis].set_color('red')
    ax.set_title("Test Image",  fontsize=10)

    # Plot retrieval images
    for i, img_res in enumerate(imgs_retrieval):
        ax = plt.subplot(1, n_retrieval, i + 2)
        img_res = cv2.resize(img_res, image_dims[:2])
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        plt.imshow(img_res)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
            ax.spines[axis].set_color('blue')
        ax.set_title("NN #%d" % (i+1), fontsize=10)
    plt.annotate('Predicted Tags: '+tags, (0,0), (-715, -10), xycoords='axes fraction', 
                 textcoords='offset points', va='top',
                 bbox=dict(fill=False, edgecolor='black', linewidth=1), size=10)
    plt.annotate('Predicted Caption: '+cap[0], (0,0), (-715, -30), xycoords='axes fraction', 
                 textcoords='offset points', va='top',
                 bbox=dict(fill=False, edgecolor='black', linewidth=1), size=10)
    plt.annotate('True Caption: '+cap[1], (0,0), (-715, -50), xycoords='axes fraction', 
                 textcoords='offset points', va='top',
                 bbox=dict(fill=False, edgecolor='black', linewidth=1), size=10)
    plt.annotate('Cosine Score: '+cap[2], (0,0), (-715, -70), xycoords='axes fraction', 
                 textcoords='offset points', va='top',
                 bbox=dict(fill=False, edgecolor='black', linewidth=1), size=10)
    if outFile is None:
        #plt.show()
        pass
    else:
        plt.savefig(outFile, bbox_inches='tight')
        plt.show()
    plt.close()
    
    
    
def countWordPlot(source, x_val, y_val, category):
    files = list(np.unique(source[y_val]))
    categories = list(np.unique(source[category]))
    color_scale = alt.Scale(domain=categories)
    base = alt.Chart(source).mark_circle().encode(
        x=x_val,
        y='count('+x_val+')',
        size='count('+x_val+')',
        color=alt.Color(category+':N', scale=color_scale),
        tooltip=[x_val]
    ).interactive()


    # A dropdown filter
    img_dropdown = alt.binding_select(options=files)
    img_select = alt.selection_single(fields=[y_val], bind=img_dropdown, name="Test")

    filter_imgs = base.add_selection(
        img_select
    ).transform_filter(
        img_select
    ).properties(title="Word Distribution of Predicted Captions", width=800, height=300 )
    return filter_imgs
    
    
def category_word_dist(source, word, category, count):
    selector = alt.selection_single(empty='all', fields=[word])
    color_scale = alt.Scale(domain=list(np.unique(source[category])))
    base = alt.Chart(source).add_selection(selector)

    points = base.mark_circle().encode(
        x=word+':O',
        y='sum('+count+'):Q',
        size='sum('+count+'):Q',
        tooltip=[word, count]
    ).properties(
        width=800,
        height=300
    )

    hists = base.mark_bar(opacity=0.8, thickness=100).encode(
        y=alt.Y(category+':O'),
        x=alt.X(count,
                stack=None,
                scale=alt.Scale()),
        color=alt.Color(category+':N',
                        scale=color_scale)
    ).transform_filter(
        selector
    ).properties(
        width=800
    )
    return points & hists



def catCountWordPlot(source, x_val, y_val, category):
    categories = list(np.unique(source[category]))
    color_scale = alt.Scale(domain=categories)
    base = alt.Chart(source).mark_circle().encode(
        x=alt.X(x_val,
                stack=None,
                scale=alt.Scale(),
                axis=alt.Axis(labelOverlap=True)),
        y='sum('+y_val+')',
        size='sum('+y_val+')',
        color=alt.Color(category+':N', scale=color_scale),
        tooltip=[x_val]
    ).interactive()


    # A dropdown filter
    img_dropdown = alt.binding_select(options=categories)
    img_select = alt.selection_single(fields=[category], bind=img_dropdown, name="Category")

    filter_imgs = base.add_selection(
        img_select
    ).transform_filter(
        img_select
    ).properties(title="Word Distribution of Predicted Captions", width=800, height=300 )
    return filter_imgs