from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Dense, LSTM, MaxoutDense, Dropout, GaussianNoise, Convolution1D, GlobalMaxPooling1D, Flatten, \
    TimeDistributed
from keras.optimizers import Adam
from keras.regularizers import l2


def embeddings_layer(max_length, embeddings, trainable=False, masking=False, scale=False, normalize=False):
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0., consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, consume_less=consume_less, dropout_U=dropout_U,
               W_regularizer=l2(l2_reg))
    if bi:
        rnn = Bidirectional(rnn)
    return rnn


def humor_RNN(wv, sent_length, **params):
    rnn_size = params.get("rnn_size", 50)
    rnn_drop_U = params.get("rnn_drop_U", 0.2)
    noise_words = params.get("noise_words", 0.3)
    drop_words = params.get("drop_words", 0.3)
    drop_sent = params.get("drop_sent", 0.3)
    sent_dense = params.get("sent_dense", 25)
    final_size = params.get("final_size", 25)
    drop_final = params.get("drop_final", 0.3)
    activity_l2 = params.get("activity_l2", 0.0001)

    ###################################################
    # Shared Layers
    ###################################################
    embedding = embeddings_layer(max_length=sent_length, embeddings=wv, masking=True)

    use_attention = True
    use_sent_dense = False

    encoder = get_RNN(LSTM, rnn_size, bi=True, return_sequences=use_attention, dropout_U=rnn_drop_U, l2_reg=0.)
    # attention = Attention()
    attention = AttentionWithContext()
    sent_dense = Dense(sent_dense)
    # sent_dense = MaxoutDense(sent_dense)
    # sent_dense = Dense(sent_dense)
    final_dense = Dense(final_size, activation="tanh")
    # sent_dense = Highway()
    # final_dense = MaxoutDense(final_size)
    time_dist = TimeDistributed(Dense(32, activation="tanh"))

    def siamese_emb(input_x):
        emb = embedding(input_x)
        # emb = BatchNormalization()(emb)
        emb = GaussianNoise(noise_words)(emb)
        emb = Dropout(drop_words)(emb)
        return emb

    def siamese_enc_att(emb):
        enc = encoder(emb)
        enc = Dropout(drop_sent)(enc)
        if use_attention:
            enc = attention(enc)
        if use_sent_dense:
            enc = sent_dense(enc)
            enc = Dropout(drop_sent)(enc)
        return enc

    def siamese_enc_unrolled(emb):
        enc = encoder(emb)
        enc = Dropout(drop_sent)(enc)
        enc = time_dist(enc)
        enc = Flatten()(enc)
        enc = Dropout(drop_sent)(enc)
        enc = sent_dense(enc)
        return enc

    ###################################################
    # Input A
    ###################################################
    input_a = Input(shape=[sent_length], dtype='int32')
    emb_a = siamese_emb(input_a)
    enc_a = siamese_enc_att(emb_a)

    ###################################################
    # Input B
    ###################################################
    input_b = Input(shape=[sent_length], dtype='int32')
    emb_b = siamese_emb(input_b)
    enc_b = siamese_enc_att(emb_b)

    ###################################################
    # Comparison
    ###################################################
    comparison = merge([enc_a, enc_b], mode='concat')
    # comparison = Flatten()(comparison)
    comparison = final_dense(comparison)
    comparison = Dropout(drop_final)(comparison)

    probabilities = Dense(1, activation='sigmoid', activity_regularizer=l2(activity_l2))(comparison)
    model = Model(input=[input_a, input_b], output=probabilities)

    # sgd = SGD(lr=0.01, clipnorm=1, momentum=True)
    # model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["binary_accuracy"])
    model.compile(optimizer=Adam(clipnorm=1, lr=0.001), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model


def humor_CNN(wv, sent_length, **params):
    filters = params.get("filters", 100)
    filter_length = params.get("filter_length ", 4)
    noise_words = params.get("noise_words", 0.3)
    drop_words = params.get("drop_words", 0.3)
    dense_size = params.get("dense_size", 50)
    drop_dense = params.get("drop_dense", 0.3)
    final_size = params.get("final_size", 50)
    drop_final = params.get("drop_final", 0.5)

    ###################################################
    # Shared Layers
    ###################################################
    embedding = embeddings_layer(max_length=sent_length, embeddings=wv, masking=False)
    encoder = Convolution1D(nb_filter=filters, filter_length=filter_length, border_mode='valid', activation='relu')
    dense = Dense(dense_size)

    ###################################################
    # Input A
    ###################################################
    input_a = Input(shape=[sent_length], dtype='int32')
    # embed sentence A
    emb_a = embedding(input_a)
    emb_a = GaussianNoise(noise_words)(emb_a)
    emb_a = Dropout(drop_words)(emb_a)
    # encode sentence A
    enc_a = encoder(emb_a)
    enc_a = GlobalMaxPooling1D()(enc_a)
    enc_a = dense(enc_a)
    enc_a = Dropout(drop_dense)(enc_a)

    ###################################################
    # Input B
    ###################################################
    input_b = Input(shape=[sent_length], dtype='int32')
    # embed sentence B
    emb_b = embedding(input_b)
    emb_b = GaussianNoise(noise_words)(emb_b)
    emb_b = Dropout(drop_words)(emb_b)
    # encode sentence B
    enc_b = encoder(emb_b)
    enc_b = GlobalMaxPooling1D()(enc_b)
    enc_b = dense(enc_b)
    enc_b = Dropout(drop_dense)(enc_b)

    ###################################################
    # Comparison
    ###################################################
    comparison = merge([enc_a, enc_b], mode='concat')
    comparison = MaxoutDense(final_size)(comparison)
    comparison = Dropout(drop_final)(comparison)

    probabilities = Dense(1, activation='sigmoid')(comparison)
    model = Model(input=[input_a, input_b], output=probabilities)

    model.compile(optimizer=Adam(clipnorm=1., lr=0.001), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model


def humor_FFNN(wv, sent_length, **params):
    noise_words = params.get("noise_words", 0.3)
    drop_words = params.get("drop_words", 0.3)

    denseA_size = params.get("denseA_size", 200)
    denseB_size = params.get("denseB_size", 100)
    denseC_size = params.get("denseC_size", 50)
    drop_dense = params.get("drop_dense", 0.3)

    final_size = params.get("final_size", 50)
    drop_final = params.get("drop_final", 0.3)

    ###################################################
    # Shared Layers
    ###################################################
    embedding = embeddings_layer(max_length=sent_length, embeddings=wv, masking=False)
    denseA = Dense(denseA_size, activation="relu")
    denseB = Dense(denseB_size, activation="relu")
    denseC = Dense(denseC_size, activation="relu")

    ###################################################
    # Input A
    ###################################################
    input_a = Input(shape=[sent_length], dtype='int32')
    # embed sentence A
    emb_a = embedding(input_a)
    emb_a = GaussianNoise(noise_words)(emb_a)
    emb_a = Dropout(drop_words)(emb_a)

    # encode sentence A
    enc_a = denseA(emb_a)
    enc_a = Dropout(drop_dense)(enc_a)
    enc_a = denseB(enc_a)
    enc_a = Dropout(drop_dense)(enc_a)
    enc_a = denseC(enc_a)
    enc_a = Dropout(drop_dense)(enc_a)

    ###################################################
    # Input B
    ###################################################
    input_b = Input(shape=[sent_length], dtype='int32')
    # embed sentence B
    emb_b = embedding(input_b)
    emb_b = GaussianNoise(noise_words)(emb_b)
    emb_b = Dropout(drop_words)(emb_b)
    # encode sentence B
    enc_b = denseA(emb_b)
    enc_b = Dropout(drop_dense)(enc_b)
    enc_b = denseB(enc_b)
    enc_b = Dropout(drop_dense)(enc_b)
    enc_b = denseC(enc_b)
    enc_b = Dropout(drop_dense)(enc_b)

    ###################################################
    # Comparison
    ###################################################
    comparison = merge([enc_a, enc_b], mode='concat')
    comparison = Flatten()(comparison)
    comparison = MaxoutDense(final_size)(comparison)
    comparison = Dropout(drop_final)(comparison)

    probabilities = Dense(1, activation='sigmoid')(comparison)
    model = Model(input=[input_a, input_b], output=probabilities)

    model.compile(optimizer=Adam(clipnorm=1., lr=0.001), loss='binary_crossentropy', metrics=["binary_accuracy"])
    return model
