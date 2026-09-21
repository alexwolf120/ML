import numpy as np

from data_utils import load_data
from layers import LinearLayer, RBFLayer, Residual
from models import Model, run_model, plot_results


FILE = "data.csv"
EPOCHS = 200
EPOCHS_RBF = 400


def _dims(x_train, y_train):
    return x_train.shape[1], int(np.max(y_train)) + 1


def experiment_linear_models(x_train, x_test, y_train, y_test):
    d, k = _dims(x_train, y_train)


    for act in ("identity", "relu", "tanh"):
        models = [
            Model([LinearLayer(d, 4, act),
                   LinearLayer(4, k, "identity")],
                  f"1L xxs ({d}-4-{k})", "tiny"),
            Model([LinearLayer(d, 16, act),
                   LinearLayer(16, k, "identity")],
                  f"1L narrow ({d}-16-{k})", "narrow"),
            Model([LinearLayer(d, 64, act),
                   LinearLayer(64, k, "identity")],
                  f"1L wide ({d}-64-{k})", "wide"),
            Model([LinearLayer(d, 32, act),
                   LinearLayer(32, 32, act),
                   LinearLayer(32, k, "identity")],
                  f"2L square ({d}-32-32-{k})", "square-2"),
            Model([LinearLayer(d, 128, act),
                   LinearLayer(128, 16, act),
                   LinearLayer(16, k, "identity")],
                  f"2L funnel ({d}-128-16-{k})", "funnel-2"),
            Model([LinearLayer(d, 16, act),
                   LinearLayer(16, 128, act),
                   LinearLayer(128, k, "identity")],
                  f"2L expand ({d}-16-128-{k})", "expand-2"),
            Model([LinearLayer(d, 32, act),
                   LinearLayer(32, 32, act),
                   LinearLayer(32, 32, act),
                   LinearLayer(32, k, "identity")],
                  f"3L deep ({d}-32-32-32-{k})", "deep-3"),
            Model([LinearLayer(d, 128, act),
                   LinearLayer(128, 64, act),
                   LinearLayer(64, 32, act),
                   LinearLayer(32, k, "identity")],
                  f"3L pyramid ({d}-128-64-32-{k})", "pyramid-3"),
            Model([LinearLayer(d, 64, act),
                   LinearLayer(64, 32, act),
                   LinearLayer(32, 16, act),
                   LinearLayer(16, 8, act),
                   LinearLayer(8, k, "identity")],
                  f"4L pyramid ({d}-64-32-16-8-{k})", "pyramid-4"),
            Model([LinearLayer(d, 32, act),
                   Residual(LinearLayer(32, 32, act)),
                   Residual(LinearLayer(32, 32, act)),
                   LinearLayer(32, k, "identity")],
                  f"residual x2 ({d}-32-wrap-wrap-{k})", "res-2"),
        ]
        results = [run_model(m, x_train, x_test, y_train, y_test,
                             epochs=EPOCHS, batch_size=32, eval_every=5)
                   for m in models]
        plot_results(results,
                     f"Linear models — activation {act}",
                     f"fig_linear_{act}")


def experiment_rbf_models(x_train, x_test, y_train, y_test):
    d, k = _dims(x_train, y_train)

    models = [
        Model([RBFLayer(d, 4, 1.0),
               RBFLayer(4, k, 1.0)],
              f"1L xxs ({d}-4-{k})", "tiny"),
        Model([RBFLayer(d, 16, 1.0),
               RBFLayer(16, k, 1.0)],
              f"1L narrow ({d}-16-{k})", "narrow"),
        Model([RBFLayer(d, 64, 1.0),
               RBFLayer(64, k, 1.0)],
              f"1L wide ({d}-64-{k})", "wide"),
        Model([RBFLayer(d, 32, 2.0),
               RBFLayer(32, k, 2.0)],
              f"1L narrow kernel ({d}-32-{k})", "sharp-kernel"),
        Model([RBFLayer(d, 32, 0.5),
               RBFLayer(32, k, 0.5)],
              f"1L wide kernel ({d}-32-{k})", "soft-kernel"),
        Model([RBFLayer(d, 32, 1.0),
               RBFLayer(32, 16, 1.0),
               RBFLayer(16, k, 1.0)],
              f"2L funnel ({d}-32-16-{k})", "funnel-2"),
        Model([RBFLayer(d, 16, 1.0),
               RBFLayer(16, 32, 0.5),
               RBFLayer(32, k, 1.0)],
              f"2L expand ({d}-16-32-{k})", "expand-2"),
        Model([RBFLayer(d, 32, 1.0),
               RBFLayer(32, 16, 0.5),
               RBFLayer(16, 8, 0.5),
               RBFLayer(8, k, 1.0)],
              f"3L pyramid ({d}-32-16-8-{k})", "pyramid-3"),
        Model([RBFLayer(d, 32, 1.0),
               RBFLayer(32, 16, 0.5),
               RBFLayer(16, 16, 0.5),
               RBFLayer(16, 8, 0.5),
               RBFLayer(8, k, 1.0)],
              f"4L pyramid ({d}-32-16-16-8-{k})", "pyramid-4"),
        Model([RBFLayer(d, 32, 1.0),
               Residual(RBFLayer(32, 32, 0.5)),
               RBFLayer(32, k, 1.0)],
              f"residual ({d}-32-wrap-{k})", "res"),
    ]
    results = [run_model(m, x_train, x_test, y_train, y_test,
                         epochs=EPOCHS_RBF, batch_size=16, eval_every=10)
               for m in models]
    plot_results(results, "RBF models (10 variants)", "fig_rbf")


def experiment_hybrid_models(x_train, x_test, y_train, y_test):
    d, k = _dims(x_train, y_train)

    models = [
        Model([LinearLayer(d, 32, "relu"),
               LinearLayer(32, k, "identity")],
              f"1L linear ({d}-32-{k})", "linear"),
        Model([RBFLayer(d, 32, 1.0),
               LinearLayer(32, k, "identity")],
              f"1L rbf ({d}-32-{k})", "rbf"),
        Model([LinearLayer(d, 32, "relu"),
               RBFLayer(32, 32, 1.0),
               LinearLayer(32, k, "identity")],
              f"2L linear-rbf ({d}-32-32-{k})", "linear-rbf"),
        Model([RBFLayer(d, 32, 1.0),
               LinearLayer(32, 32, "relu"),
               LinearLayer(32, k, "identity")],
              f"2L rbf-linear ({d}-32-32-{k})", "rbf-linear"),
        Model([LinearLayer(d, 16, "relu"),
               RBFLayer(16, 64, 1.0),
               LinearLayer(64, k, "identity")],
              f"2L expand ({d}-16-64-{k})", "linear-rbf-big"),
        Model([RBFLayer(d, 64, 1.0),
               LinearLayer(64, 16, "relu"),
               LinearLayer(16, k, "identity")],
              f"2L funnel ({d}-64-16-{k})", "rbf-linear-small"),
        Model([LinearLayer(d, 32, "relu"),
               RBFLayer(32, 32, 1.0),
               LinearLayer(32, k, "identity")],
              f"3L linear-rbf-linear ({d}-32-32-{k})", "linear-rbf-linear"),
        Model([RBFLayer(d, 32, 1.0),
               LinearLayer(32, 32, "relu"),
               RBFLayer(32, k, 1.0)],
              f"3L rbf-linear-rbf ({d}-32-32-{k})", "rbf-linear-rbf"),
        Model([LinearLayer(d, 64, "relu"),
               RBFLayer(64, 32, 1.0),
               RBFLayer(32, 16, 0.5),
               LinearLayer(16, k, "identity")],
              f"4L linear-rbf-rbf-linear ({d}-64-32-16-{k})",
              "linear-rbf-rbf-linear"),
        Model([RBFLayer(d, 32, 1.0),
               Residual(LinearLayer(32, 32, "relu")),
               Residual(LinearLayer(32, 32, "relu")),
               LinearLayer(32, k, "identity")],
              f"residual x2 ({d}-32rbf-wrap-wrap-{k})", "rbf-res-2"),
    ]
    results = [run_model(m, x_train, x_test, y_train, y_test,
                         epochs=EPOCHS, batch_size=32, eval_every=5)
               for m in models]
    plot_results(results, "Hybrid models", "fig_hybrid")


def do_experiments():
    x_train, x_test, y_train, y_test = load_data(FILE)
    experiment_linear_models(x_train, x_test, y_train, y_test)
    experiment_rbf_models(x_train, x_test, y_train, y_test)
    experiment_hybrid_models(x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    do_experiments()