def create_dataset():
    labels = np.array([[0, 0, 0],
                       [0, 0, 1],
                       [0, 1, 2],
                       [0, 1, 3],
                       [0, 1, 4],
                       [1, 2, 5],
                       [1, 2, 6],
                       [1, 2, 7],
                       [1, 3, 8],
                       [1, 3, 9],
                       [1, 3, 10]], dtype=np.int32)
    n_samples = 1000
    y_indices = np.random.randint(11, size=n_samples)
    y = labels[y_indices]
    x = np.random.randn(n_samples, 1000)
    x = x.astype(np.float32)
    return x, y, labels


def run():
    dropout_rate = 0.5
    latent_dim = 50

    x, y, label_pool = create_dataset()
    d = np.ones(y.shape[0])
    y_oh = to_categorical(y[:, -1])

    n_features = x.shape[1]

    model = make_model(n_features, latent_dim, dropout_rate,
                       label_pool)
    model.fit(x=[x, y, d], y=y_oh, epochs=50)
    loss = model.evaluate(x=[x, y, d], y=y_oh)
    print()
    print(loss)