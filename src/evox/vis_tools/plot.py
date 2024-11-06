import jax.numpy as jnp

from evox import use_state


def plot_dec_space(
    population_history,
    **kwargs,
):
    """用于可视化决策空间中种群的演化过程
    
    该函数创建一个交互式动画,展示种群在决策空间中的分布变化。动画包含播放控制和进度条。
    
    参数:
        population_history: 列表,包含每一代的种群位置数据,每个元素shape为(pop_size, 2)
        **kwargs: 传递给plotly布局的额外参数,用于自定义图表样式
        
    返回:
        plotly.graph_objects.Figure对象
        
    注意:
        - 需要安装plotly库
        - 仅支持2D决策空间的可视化
        - 自动计算适当的坐标轴范围,并留有10%的边界空间
    """
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")

    all_pop = jnp.concatenate(population_history, axis=0)
    x_lb = jnp.min(all_pop[:, 0])
    x_ub = jnp.max(all_pop[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.1 * x_range
    x_ub = x_ub + 0.1 * x_range
    y_lb = jnp.min(all_pop[:, 1])
    y_ub = jnp.max(all_pop[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.1 * y_range
    y_ub = y_ub + 0.1 * y_range

    frames = []
    steps = []
    for i, pop in enumerate(population_history):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=pop[:, 0],
                        y=pop[:, 1],
                        mode="markers",
                        marker={"color": "#636EFA"},
                    ),
                ],
                name=str(i),
            )
        )
        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"t": 50},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [x_lb, x_ub]},
            yaxis={"range": [y_lb, y_ub]},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 200,
                                        "easing": "linear",
                                    },
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig


def plot_obj_space_1d(fitness_history, animation=True, **kwargs):
    """绘制一维目标空间的进化过程
    
    该函数可以创建静态或动态的适应度变化图表,展示种群适应度的统计信息随代数的变化。
    
    参数:
        fitness_history: 列表,包含每一代的适应度值
        animation: 布尔值,是否创建动画效果
        **kwargs: 传递给plotly布局的额外参数
    
    返回:
        plotly.graph_objects.Figure对象
    """
    if animation:
        return plot_obj_space_1d_animation(fitness_history, **kwargs)
    else:
        return plot_obj_space_1d_no_animation(fitness_history, **kwargs)


def plot_obj_space_1d_no_animation(fitness_history, **kwargs):
    """绘制一维目标空间的静态统计图表
    
    该函数创建一个包含最小值、最大值、中位数和平均值的多线图,
    用于展示种群适应度随代数的变化趋势。
    
    参数:
        fitness_history: 列表,包含每一代的适应度值数组
        **kwargs: 传递给plotly布局的额外参数
        
    返回:
        plotly.graph_objects.Figure对象
        
    注意:
        - 需要安装plotly库
        - 适用于单目标优化问题
        - 生成静态图表,不包含动画效果
    """
    # 检查plotly依赖是否安装
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")

    # 计算每一代的统计指标
    min_fitness = [jnp.min(x) for x in fitness_history]    # 每代的最小适应度
    max_fitness = [jnp.max(x) for x in fitness_history]    # 每代的最大适应度
    median_fitness = [jnp.median(x) for x in fitness_history]  # 每代的中位数
    avg_fitness = [jnp.mean(x) for x in fitness_history]   # 每代的平均值
    generation = jnp.arange(len(fitness_history))          # 代数序列

    # 创建多线图对象
    fig = go.Figure(
        [
            # 绘制四条统计曲线
            go.Scatter(x=generation, y=min_fitness, mode="lines", name="Min"),      # 最小值曲线
            go.Scatter(x=generation, y=max_fitness, mode="lines", name="Max"),      # 最大值曲线
            go.Scatter(x=generation, y=median_fitness, mode="lines", name="Median"),# 中位数曲线
            go.Scatter(x=generation, y=avg_fitness, mode="lines", name="Average"),  # 平均值曲线
        ],
        # 配置图表布局
        layout=go.Layout(
            # 设置图例位置在右上角
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
                "xanchor": "auto",
            },
            # 移除图表边距,使图形充满整个区域
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
        ),
    )

    return fig


def plot_obj_space_1d_animation(fitness_history, **kwargs):
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")

    min_fitness = [jnp.min(x) for x in fitness_history]
    max_fitness = [jnp.max(x) for x in fitness_history]
    median_fitness = [jnp.median(x) for x in fitness_history]
    avg_fitness = [jnp.mean(x) for x in fitness_history]
    generation = jnp.arange(len(fitness_history))

    frames = []
    steps = []
    for i in range(len(fitness_history)):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=generation[: i + 1],
                        y=min_fitness[: i + 1],
                        mode="lines",
                        name="Min",
                        showlegend=True,
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=max_fitness[: i + 1],
                        mode="lines",
                        name="Max",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=median_fitness[: i + 1],
                        mode="lines",
                        name="Median",
                    ),
                    go.Scatter(
                        x=generation[: i + 1],
                        y=avg_fitness[: i + 1],
                        mode="lines",
                        name="Average",
                    ),
                ],
                name=str(i),
            )
        )

        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 1, "t": 10},
            "len": 0.8,
            "x": 0.2,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    lb = min(min_fitness)
    ub = max(max_fitness)
    fit_range = ub - lb
    lb = lb - 0.05 * fit_range
    ub = ub + 0.05 * fit_range
    fig = go.Figure(
        data=frames[-1].data,
        layout=go.Layout(
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
                "xanchor": "auto",
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            xaxis={"range": [0, len(fitness_history)], "autorange": False},
            yaxis={"range": [lb, ub], "autorange": False},
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig


def plot_obj_space_2d(fitness_history, problem_pf=None, sort_points=False, **kwargs):
    """绘制二维目标空间的进化过程动画
    
    该函数创建一个交互式动画,展示种群在二维目标空间中的分布变化。
    可以选择性地显示问题的真实帕累托前沿作为参考。
    
    参数:
        fitness_history: 列表,包含每一代的适应度值矩阵,每个矩阵shape为(pop_size, 2)
        problem_pf: 可选,问题的真实帕累托前沿点集,shape为(n_points, 2)
        sort_points: 布尔值,是否对每代的点进行排序以获得更平滑的动画效果
        **kwargs: 传递给plotly布局的额外参数
        
    返回:
        plotly.graph_objects.Figure对象
        
    注意:
        - 需要安装plotly库
        - 仅支持双目标优化问题的可视化
        - 自动计算适当的坐标轴范围,并留有5%的边界空间
    """
    # 检查plotly依赖
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")

    # 计算整体坐标范围并添加5%的边界空间
    all_fitness = jnp.concatenate(fitness_history, axis=0)  # 合并所有代的适应度
    # 计算x轴范围
    x_lb = jnp.min(all_fitness[:, 0])
    x_ub = jnp.max(all_fitness[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.05 * x_range  # 添加5%的左边界
    x_ub = x_ub + 0.05 * x_range  # 添加5%的右边界
    # 计算y轴范围
    y_lb = jnp.min(all_fitness[:, 1])
    y_ub = jnp.max(all_fitness[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.05 * y_range  # 添加5%的下边界
    y_ub = y_ub + 0.05 * y_range  # 添加5%的上边界

    # 准备动画帧和控制步骤
    frames = []
    steps = []
    
    # 如果提供了帕累托前沿,创建其散点图对象
    if problem_pf is not None:
        pf_scatter = go.Scatter(
            x=problem_pf[:, 0],
            y=problem_pf[:, 1],
            mode="markers",
            marker={"color": "#FFA15A", "size": 2},  # 使用橙色标记真实帕累托前沿
            name="Pareto Front",
        )

    # 为每一代创建动画帧
    for i, fit in enumerate(fitness_history):
        # 可选择对点进行排序以获得更平滑的动画效果
        if sort_points:
            indices = jnp.lexsort(fit.T)  # 按照目标值排序
            fit = fit[indices]
            
        # 创建当前代的种群散点图
        scatter = go.Scatter(
            x=fit[:, 0],
            y=fit[:, 1],
            mode="markers",
            marker={"color": "#636EFA"},  # 使用蓝色标记种群
            name="Population",
        )
        
        # 将帕累托前沿(如果有)和种群点添加到动画帧
        if problem_pf is not None:
            frames.append(go.Frame(data=[pf_scatter, scatter], name=str(i)))
        else:
            frames.append(go.Frame(data=[scatter], name=str(i)))

        # 创建动画控制步骤
        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 200},
                },
            ],
        }
        steps.append(step)

    # 配置进度条
    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},  # 显示当前代数
            "pad": {"b": 1, "t": 10},
            "len": 0.8,  # 进度条长度
            "x": 0.2,    # 进度条位置
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    
    # 创建最终的图表对象
    fig = go.Figure(
        data=frames[0].data,  # 初始显示第一代数据
        layout=go.Layout(
            # 配置图例位置
            legend={
                "x": 1,
                "y": 1,
                "xanchor": "auto",
                "xanchor": "auto",
            },
            # 移除图表边距
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            # 添加进度条
            sliders=sliders,
            # 设置坐标轴范围
            xaxis={"range": [x_lb, x_ub], "autorange": False},
            yaxis={"range": [y_lb, y_ub], "autorange": False},
            # 配置动画控制按钮
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        # 播放按钮
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {
                                        "duration": 200,
                                        "easing": "linear",
                                    },
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        # 暂停按钮
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    # 按钮位置和样式设置
                    "x": 0.2,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 30},
                },
            ],
            **kwargs,  # 合并用户自定义的布局参数
        ),
        frames=frames,  # 添加所有动画帧
    )

    return fig


def plot_obj_space_3d(fitness_history, sort_points=False, problem_pf=None, **kwargs):
    try:
        import plotly
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("The plot function requires plotly to be installed.")

    all_fitness = jnp.concatenate(fitness_history, axis=0)
    x_lb = jnp.min(all_fitness[:, 0])
    x_ub = jnp.max(all_fitness[:, 0])
    x_range = x_ub - x_lb
    x_lb = x_lb - 0.05 * x_range
    x_ub = x_ub + 0.05 * x_range

    y_lb = jnp.min(all_fitness[:, 1])
    y_ub = jnp.max(all_fitness[:, 1])
    y_range = y_ub - y_lb
    y_lb = y_lb - 0.05 * y_range
    y_ub = y_ub + 0.05 * y_range

    z_lb = jnp.min(all_fitness[:, 2])
    z_ub = jnp.max(all_fitness[:, 2])
    z_range = z_ub - z_lb
    z_lb = z_lb - 0.05 * z_range
    z_ub = z_ub + 0.05 * z_range

    frames = []
    steps = []
    for i, fit in enumerate(fitness_history):
        # it will make the animation look nicer
        if sort_points:
            indices = jnp.lexsort(fit.T)
            fit = fit[indices]

        scatter = go.Scatter3d(
            x=fit[:, 0],
            y=fit[:, 1],
            z=fit[:, 2],
            mode="markers",
            marker={"color": "#636EFA", "size": 2},
        )
        frames.append(go.Frame(data=[scatter], name=str(i)))

        step = {
            "label": i,
            "method": "animate",
            "args": [
                [str(i)],
                {
                    "frame": {"duration": 200},
                    "mode": "immediate",
                    "transition": {"duration": 200, "easing": "linear"},
                },
            ],
        }
        steps.append(step)

    sliders = [
        {
            "currentvalue": {"prefix": "Generation: "},
            "pad": {"b": 10, "t": 50},
            "len": 0.5,
            "x": 0.3,
            "y": 0,
            "yanchor": "top",
            "xanchor": "left",
            "steps": steps,
        }
    ]
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            sliders=sliders,
            scene={
                "xaxis": {"range": [x_lb, x_ub], "autorange": False},
                "yaxis": {"range": [y_lb, y_ub], "autorange": False},
                "zaxis": {"range": [z_lb, z_ub], "autorange": False},
                "aspectmode": "cube",
            },
            scene_camera={
                "eye": {"x": 2, "y": 0.5, "z": 0.5},
            },
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 200},
                                    "fromcurrent": True,
                                },
                            ],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [
                                [None],
                                {
                                    "frame": {"duration": 0},
                                    "mode": "immediate",
                                },
                            ],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "x": 0.3,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top",
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                },
            ],
            **kwargs,
        ),
        frames=frames,
    )

    return fig
