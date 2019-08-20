import numpy
from matplotlib import pyplot
import careovoys.multi

__all__ = ['plot_cohorts']


_models = {
    'kaplan-meier': lambda ci: careovoys.multi.KaplanMeier(),
    'exponential': lambda ci: careovoys.multi.Exponential(ci=ci),
    'weibull': lambda ci: careovoys.multi.Weibull(ci=ci),
    'gamma': lambda ci: careovoys.multi.Gamma(ci=ci),
    'generalized-gamma': lambda ci: careovoys.multi.GeneralizedGamma(ci=ci),
}


def plot_cohorts(G, B, T, target='retention', t_max=None, model='kaplan-meier',
                 ci=None, ax=None, plot_kwargs={}, plot_ci_kwargs={},
                 groups=None, specific_groups=None, limited=True,
                 label_fmt='%(group)s (n=%(n).0f, k=%(k).0f)'):
    ''' Helper function to fit data using a model and then plot the cohorts.

    :param G: list with group assignment (see :meth:`utils.get_arrays`)
    :param B: list with group assignment (see :meth:`utils.get_arrays`)
    :param T: list with group assignment (see :meth:`utils.get_arrays`)
    :param target: target variable (either retention or conversion) to plot
    :param t_max: (optional) max value for x axis
    :param model: (optional, default is kaplan-meier) model to fit.
        Can be an instance of :class:`multi.MultiModel` or a string
        identifying the model. One of 'kaplan-meier', 'exponential',
        'weibull', 'gamma', or 'generalized-gamma'.
    :param ci: confidence interval, value from 0-1, or None (default) if
        no confidence interval is to be plotted
    :param ax: custom pyplot axis to plot on
    :param plot_kwargs: extra arguments to pyplot for the lines
    :param plot_ci_kwargs: extra arguments to pyplot for the confidence
        intervals
    :param groups: list of group labels
    :param specific_groups: subset of groups to plot
    :param label_fmt: custom format for the labels to use in the legend
    '''

    if model not in _models.keys():
        if not isinstance(model, careovoys.multi.MultiModel):
            raise Exception('model incorrectly specified')

    if groups is None:
        groups = list(set(G))

    if ax is None:
        ax = pyplot.gca()

    # Set x scale
    if t_max is None:
        _, t_max = ax.get_xlim()
        t_max = max(t_max, 2. * max(T))
    if not isinstance(model, careovoys.multi.MultiModel):
        # Fit model
        m = _models[model](ci=bool(ci))
        m.fit(G, B, T)
    else:
        m = model

    if specific_groups is None:
        specific_groups = groups

    if len(set(specific_groups).intersection(groups)) != len(specific_groups):
        raise Exception('specific_groups not a subset of groups!')

    # Plot
    t = numpy.linspace(0, t_max, 1000)
    _, y_max = ax.get_ylim()
    ax.set_prop_cycle(None)  # Reset to first color
    for i, group in enumerate(specific_groups):
        print(group)
        j = groups.index(group)  # matching index of group

        n = sum(1 for g in G if g == j)  # TODO: slow
        k = sum(1 for g, b in zip(G, B) if g == j and b)  # TODO: slow
        label = label_fmt % dict(group=group, n=n, k=k)

        if ci is not None:
            if limited:
                p_y, p_y_lo, p_y_hi = m.cdf(j, t, ci=ci).T
            else:
                p_y, p_y_lo, p_y_hi = m.cdf(j, t, ci=ci, limited=False).T
            merged_plot_ci_kwargs = {'alpha': 0.2}
            merged_plot_ci_kwargs.update(plot_ci_kwargs)
            if target == 'retention':
                p = ax.fill_between(t, 100. * (1. - p_y_lo), 100. * (1. - p_y_hi),
                                    **merged_plot_ci_kwargs)
            else:
                p = ax.fill_between(t, 100. * p_y_lo, 100. * p_y_hi,
                                    **merged_plot_ci_kwargs)
            color = p.get_facecolor()[0]  # reuse color for the line
        else:
            if limited:
                p_y = m.cdf(j, t).T
            else:
                p_y = m.cdf(j, t, limited=False).T
            color = None

        merged_plot_kwargs = {'color': color, 'linewidth': 2.5,
                              'alpha': 0.7}
        merged_plot_kwargs.update(plot_kwargs)
        if target == 'retention':
            ax.plot(t, 100. * (1. - p_y), label=label, **merged_plot_kwargs)
            y_max = max(y_max, 110. * max(1. - p_y))
        else:
            ax.plot(t, 100. * p_y, label=label, **merged_plot_kwargs)
            y_max = max(y_max, 110. * max(p_y))

    ax.set_xlim([0, t_max])
    ax.set_ylim([0, y_max])
    if target == 'retention':
        ax.set_ylabel('Retention rate %')
        ax.set_xlabel('Days since First Delivery')
    else:
        ax.set_ylabel('Conversion rate %')
        ax.set_xlabel('Days since Initial Survey')
    ax.grid(True)
    return m
