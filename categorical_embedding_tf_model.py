import tensorflow as tf

def build_estimator(model_dir, model_type="combined"):
    """Build an estimator."""
    # Sparse base columns.
    country = tf.contrib.layers.sparse_column_with_keys(column_name="country",
                                                       keys=["Germany", "France", "Austria"])
    promo1 = tf.contrib.layers.sparse_column_with_keys(column_name="promo1",
                                                       keys=["0", "1"])
    promo2 = tf.contrib.layers.sparse_column_with_keys(column_name="promo2",
                                                       keys=["0", "1"])
    article = tf.contrib.layers.sparse_column_with_hash_bucket(
        "article", hash_bucket_size=2000)
    productgroup = tf.contrib.layers.sparse_column_with_hash_bucket(
        "productgroup", hash_bucket_size=100)
    category = tf.contrib.layers.sparse_column_with_hash_bucket(
        "category", hash_bucket_size=1000)
    style = tf.contrib.layers.sparse_column_with_keys(column_name="style",
                                                       keys=["wide", "slim", "regular"])
    sizes = tf.contrib.layers.sparse_column_with_hash_bucket(
        "sizes", hash_bucket_size=100)
    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender",
                                                       keys=["unisex", "kids", "female", "male"])

    # Continuous base columns.
    regular_price = tf.contrib.layers.real_valued_column("regular_price")
    current_price = tf.contrib.layers.real_valued_column("current_price")
    ratio = tf.contrib.layers.real_valued_column("ratio")
    cost = tf.contrib.layers.real_valued_column("cost")
    day = tf.contrib.layers.real_valued_column("day")
    week = tf.contrib.layers.real_valued_column("week")
    month = tf.contrib.layers.real_valued_column("month")
    year = tf.contrib.layers.real_valued_column("year")
    dayofyear = tf.contrib.layers.real_valued_column("dayofyear")
    rgb_r_main_col = tf.contrib.layers.real_valued_column("rgb_r_main_col")
    rgb_g_main_col = tf.contrib.layers.real_valued_column("rgb_g_main_col")
    rgb_b_main_col = tf.contrib.layers.real_valued_column("rgb_b_main_col")
    rgb_r_sec_col = tf.contrib.layers.real_valued_column("rgb_r_sec_col")
    rgb_g_sec_col = tf.contrib.layers.real_valued_column("rgb_g_sec_col")
    rgb_b_sec_col = tf.contrib.layers.real_valued_column("rgb_b_sec_col")

    # Transformations.
    rgb_r_main_col_buckets = tf.contrib.layers.bucketized_column(rgb_r_main_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])
    rgb_g_main_col_buckets = tf.contrib.layers.bucketized_column(rgb_g_main_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])
    rgb_b_main_col_buckets = tf.contrib.layers.bucketized_column(rgb_b_main_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])
    rgb_r_sec_col_buckets = tf.contrib.layers.bucketized_column(rgb_r_sec_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])
    rgb_g_sec_col_buckets = tf.contrib.layers.bucketized_column(rgb_g_sec_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])
    rgb_b_sec_col_buckets = tf.contrib.layers.bucketized_column(rgb_b_sec_col,
                                                                 boundaries=[
                                                                     32, 64, 96, 128, 160, 192, 224])

    # Wide columns and deep columns.
    wide_columns = [country, promo1, promo2, article, productgroup,
                    category, style, sizes, gender,
                    tf.contrib.layers.crossed_column(
                        [rgb_r_main_col_buckets, rgb_g_main_col_buckets, rgb_b_main_col_buckets],
                        hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column(
                        [rgb_r_sec_col_buckets, rgb_g_sec_col_buckets, rgb_b_sec_col_buckets],
                        hash_bucket_size=int(1e6)),
                   day, week, month, year]
    deep_columns = [
        tf.contrib.layers.embedding_column(country, dimension=2),
        tf.contrib.layers.embedding_column(promo1, dimension=9),
        tf.contrib.layers.embedding_column(promo2, dimension=2),
        tf.contrib.layers.embedding_column(article, dimension=2),
        tf.contrib.layers.embedding_column(productgroup, dimension=3),
        tf.contrib.layers.embedding_column(category, dimension=5),
        tf.contrib.layers.embedding_column(style, dimension=2),
        tf.contrib.layers.embedding_column(sizes, dimension=3),
        tf.contrib.layers.embedding_column(gender, dimension=2),
        regular_price, current_price, ratio, cost,
        day, week, month, year, dayofyear,
        rgb_r_main_col, rgb_g_main_col, rgb_b_main_col,
        rgb_r_sec_col, rgb_g_sec_col, rgb_b_sec_col
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearRegressor(model_dir=model_dir,
                                          feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNRegressor(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedRegressor(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True,
        config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))
    return m