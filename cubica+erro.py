import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# ----
# Aparência com fallback se estilos específicos não existirem
# ----
preferred_styles = ['classic']
for st in preferred_styles:
    if st in plt.style.available:
        plt.style.use(st)
        break
else:
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.grid': True,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.titleweight': 'bold',
        'figure.dpi': 100
    })

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['ARIAL']
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15
# ----


def subdivisao_cubica(a, b, n, k, func=None):
    """
    Gera os níveis de subdivisão cúbica.

    a, b: intervalo dos valores x iniciais
    n: determina a quantidade de pontos iniciais (2^n + 1)
    k: número de iterações de subdivisão
    func: função para geração dos valores y iniciais (recebe o array x)

    Retorna:
      plot_x: lista de arrays x para cada nível (do nível 0 ao nível k)
      plot_y: lista de arrays y correspondentes
    """
    J = 2 ** n
    x = np.linspace(a, b, J + 1)
    if func is None:
        y = np.abs(x - np.floor(x) - 0.5)
    else:
        y = func(x)

    plot_x = [x.copy()]
    plot_y = [y.copy()]

    for it in range(k):
        xs = []
        ys = []
        N = len(x)

        # Primeiro bloco de 4 pontos
        xs.append(x[0])
        ys.append(y[0])
        x_meio = 0.5 * (x[0] + x[1])
        y_meio = (
            (5 / 16) * y[0] +
            (15 / 16) * y[1] +
            (-5 / 16) * y[2] +
            (1 / 16) * y[3]
        )
        xs.append(x_meio)
        ys.append(y_meio)

        # Blocos intermediários de 4 pontos
        for i in range(1, N - 2):
            xs.append(x[i])
            ys.append(y[i])
            x_meio = 0.5 * (x[i] + x[i + 1])
            y_meio = (
                (-1 / 16) * y[i - 1] +
                (9 / 16) * y[i] +
                (9 / 16) * y[i + 1] +
                (-1 / 16) * y[i + 2]
            )
            xs.append(x_meio)
            ys.append(y_meio)

        # Penúltimo e último blocos (forçando pontos finais)
        if N > 3:
            xs.append(x[N - 2])
            ys.append(y[N - 2])
            x_meio = 0.5 * (x[N - 2] + x[N - 1])
            y_meio = (
                (1 / 16) * y[N - 4] +
                (-5 / 16) * y[N - 3] +
                (15 / 16) * y[N - 2] +
                (5 / 16) * y[N - 1]
            )
            xs.append(x_meio)
            ys.append(y_meio)
            # Último ponto
            xs.append(x[N - 1])
            ys.append(y[N - 1])
            x_meio = 0.5 * (x[N - 2] + x[N - 1])
            y_meio = (
                (1 / 16) * y[N - 4] +
                (-5 / 16) * y[N - 3] +
                (15 / 16) * y[N - 2] +
                (5 / 16) * y[N - 1]
            )

        x = np.array(xs)
        y = np.array(ys)
        plot_x.append(x.copy())
        plot_y.append(y.copy())
    return plot_x, plot_y


def _default_func(x):
    """Função padrão quando func for None."""
    return np.abs(x - np.floor(x) - 0.5)


def _set_dynamic_log_ticks(ax, values, exp_step=2):
    """Define ticks logarítmicos dinâmicos com espaçamento fixo de exp_step décadas."""
    positive_values = np.asarray(values)
    positive_values = positive_values[positive_values > 0]

    if positive_values.size == 0:
        ticks = np.array([1e-16, 1.0])
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        return

    min_val = positive_values.min()
    max_val = positive_values.max()

    exp_min = int(np.floor(np.log10(min_val)))
    exp_max = int(np.ceil(np.log10(max_val)))

    start_exp = exp_step * int(np.floor(exp_min / exp_step))
    end_exp = exp_step * int(np.ceil(exp_max / exp_step))

    exponents = np.arange(start_exp, end_exp + exp_step, exp_step)
    ticks = 10.0 ** exponents

    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))


def plot_and_save_error_curves(plot_x, plot_y, func=None, output_dir='output_erro',
                               atol_isclose=1e-12, rtol_isclose=1e-14, linscale=0.5):
    """
    Salva gráficos de erro absoluto para cada nível (exceto nível 0).
    - Usa escala linear quando erros são negligíveis (e.g., para polinômios cúbicos)
    - Usa escala logarítmica quando há erros significativos (e.g., para seno)
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func_to_use = _default_func
    else:
        func_to_use = func

    for i, (xs, ys) in enumerate(zip(plot_x, plot_y)):
        if i == 0:
            continue

        y_true = func_to_use(xs)
        err = np.abs(ys - y_true)
        max_err = err.max()

        plt.figure(figsize=(9, 6))
        ax = plt.gca()

        # If maximum error is negligible (e.g., exact cubics), use linear scale
        if max_err < 1e-10:
            ax.plot(xs, err, linestyle='-', linewidth=0.9, color='#1f77b4', alpha=0.95)
            ax.plot(xs, err, linestyle='None', marker='o', color='#1f77b4', markersize=5)
            ax.set_ylabel('Absolute error', fontsize=13)
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            # Significant errors: use log scale
            err_plot = np.where(err == 0, err.max() * 1e-3, err)  # avoid log(0)
            ax.semilogy(xs, err_plot, linestyle='-', linewidth=0.9, color='#1f77b4', alpha=0.95)
            ax.semilogy(xs, err_plot, linestyle='None', marker='o', color='#1f77b4', markersize=5)
            ax.set_ylabel('Absolute error (log scale)', fontsize=13)
            _set_dynamic_log_ticks(ax, err_plot, exp_step=2)

        ax.set_xlabel('x', fontsize=13)
        ax.set_title(f'Absolute error - Level {i}', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

        plt.tight_layout()
        filename = os.path.join(output_dir, f'erro_iteracao_{i}.png')
        plt.savefig(filename)
        plt.close()


def plot_final_with_error(plot_x, plot_y, func=None, output_dir='output_summary',
                          dense_points=3000, atol_isclose=1e-12, rtol_isclose=1e-14, linscale=0.5):
    """
    Gera e salva uma figura resumo com:
      - Subplot superior: função contínua (densamente amostrada), aproximação final, e marcação dos nós iniciais.
      - Subplot inferior: erro absoluto nos nós finais em escala 'compressa' (symlog),
        incluindo explicitamente os nós iniciais no gráfico de erro.
    """
    os.makedirs(output_dir, exist_ok=True)
    if func is None:
        func_to_use = _default_func
    else:
        func_to_use = func

    final_x = plot_x[-1]
    final_y = plot_y[-1]
    initial_x = plot_x[0]
    initial_y = _default_func(initial_x) if func is None else func(initial_x)

    x_min, x_max = final_x.min(), final_x.max()
    xs_dense = np.linspace(x_min, x_max, dense_points)
    y_dense = func_to_use(xs_dense)

    y_all = np.concatenate([y_dense, final_y, initial_y])
    y_min, y_max = y_all.min(), y_all.max()
    common_margin = 0.05 * max(y_max - y_min, x_max - x_min)

    # erro nos nós finais
    y_true_nodes = func_to_use(final_x)
    err_nodes = np.abs(final_y - y_true_nodes)
    max_err = err_nodes.max()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8),
                                         gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Top: função contínua e aproximação final (pontos e linha)
    ax_top.plot(xs_dense, y_dense, color='#2c7fb8', linewidth=1.8, label='function')
    ax_top.plot(final_x, final_y, marker='o', color='black', markerfacecolor='#d7261e',
                linestyle='-', linewidth=0.9, label='subdivision', markersize=2)

    # Mostrar nós iniciais de forma destacada (apresenta os dados iniciais)
    ax_top.scatter(initial_x, initial_y, s=80, facecolors='none', edgecolors='#2ca02c',
                   linewidths=1.5, label='starting nodes', zorder=5)

    ax_top.set_ylabel(' ', fontsize=12)
    ax_top.set_title(' ', fontsize=14)
    ax_top.grid(False)
    ax_top.legend(fontsize=14, frameon=False, loc='best')
    ax_top.set_xlim(x_min - common_margin - 0.05, x_max + common_margin + 0.05)
    ax_top.set_ylim(y_min - common_margin, y_max + common_margin)


    # Bottom: adaptive error (linear if negligible, log if significant)
    if max_err < 1e-10:
        # Negligible error: linear scale (shows machine precision)
        ax_bot.plot(final_x, err_nodes, linestyle='None', color='#ff7f0e', alpha=0.95)
        ax_bot.plot(final_x, err_nodes, linestyle='None', marker='o', color='#ff7f0e', markersize=2)
        ax_bot.set_ylabel('Absolute error', fontsize=14)
        ax_bot.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        # Significant error: log scale
        err_plot = np.where(err_nodes == 0, max_err * 1e-3, err_nodes)
        ax_bot.semilogy(final_x, err_plot, linestyle='None', color='#ff7f0e', alpha=0.95)
        ax_bot.semilogy(final_x, err_plot, linestyle='None', marker='o', color='#ff7f0e', markersize=2)
        ax_bot.set_ylabel('Error (log scale)', fontsize=14)
        _set_dynamic_log_ticks(ax_bot, err_plot, exp_step=2)

    ax_bot.set_xlabel('x', fontsize=12)
    ax_bot.grid(False)

    # sem legenda no plot de erro (conforme pedido)
    # pronto para salvar
    plt.tight_layout()
    filename = os.path.join(output_dir, 'summary_final.png')
    plt.savefig(filename)
    plt.close()
    
sin = lambda x: np.sin(2 * np.pi * x)
cos = lambda x: np.cos(2 * np.pi * x)
quad = lambda x: x ** 2
cub = lambda x: x ** 3
quint = lambda x: x ** 5

if __name__ == '__main__':
    # --- Parâmetros padrão ---
    a = -1.0
    b = 1.0
    n = 4
    k = 8
    func = quad
    #lambda x: np.sin(2 * np.pi * x)

    # Geração dos pontos e subdivisão
    plot_x, plot_y = subdivisao_cubica(a, b, n, k, func=func)

    # Salva os gráficos de subdivisão (cada nível)
    output_dir = 'output_subdivisao_cubica'
    os.makedirs(output_dir, exist_ok=True)
    # Calculate global y-limits and x-limits for all iterations
    all_y = np.concatenate(plot_y)
    y_min, y_max = all_y.min(), all_y.max()

    all_x = np.concatenate(plot_x)
    x_min, x_max = all_x.min(), all_x.max()

    # Use the same absolute margin on both axes
    common_margin = 0.05 * max(y_max - y_min, x_max - x_min)
    y_min_plot = y_min - common_margin
    y_max_plot = y_max + common_margin
    x_min_plot = x_min - common_margin
    x_max_plot = x_max + common_margin

    for i in range(len(plot_x)):
        plt.figure(figsize=(8, 8))
        plt.plot(plot_x[i], plot_y[i], linestyle=' ', marker='o',
                 label=f'Level {i}', color='black', markeredgecolor='red',
                 markerfacecolor='red', markersize=5)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title(' ')
        plt.grid(False)
        plt.legend('', frameon=False)
        plt.xlim(x_min_plot, x_max_plot)
        plt.ylim(y_min_plot, y_max_plot)
        plt.tight_layout()
        filename = os.path.join(output_dir, f'iteracao_{i}.png')
        plt.savefig(filename)
        plt.close()
    print(f"Imagens salvas em: {output_dir}/")

    # Gera e salva gráficos de erro em escala comprimida (não inclui legenda; nós iniciais incluídos)
    plot_and_save_error_curves(plot_x, plot_y, func=func, output_dir='output_erro',
                               atol_isclose=1e-12, rtol_isclose=1e-14, linscale=0.5)
    print("Imagens de erro salvas em: output_erro/ (nível 0 omitido por convenção; nós iniciais incluídos)")

    # Gera uma figura resumo: função contínua vs subdivisão final e erro dos nós na parte inferior
    plot_final_with_error(plot_x, plot_y, func=func, output_dir='output_summary',
                          dense_points=3000, atol_isclose=1e-12, rtol_isclose=1e-14, linscale=0.5)
    print("Figura resumo (função contínua, subdivisão final e erro) salva em: output_summary/summary_final.png")