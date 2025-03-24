import numpy as np
import plotly.graph_objects as go
import os


def plot_cv_results(parameters, input_mae, model, sample_length, params_index):
    for kernel_model in parameters.keys():
        # 2D plot of MAE avg. with quantile bonds versus the parameters
        avg_mae = np.mean(input_mae[kernel_model], axis=0)
        q95 = np.quantile(input_mae[kernel_model], 0.95, axis=0)
        q05 = np.quantile(input_mae[kernel_model], 0.05, axis=0)
        q75 = np.quantile(input_mae[kernel_model], 0.75, axis=0)
        q25 = np.quantile(input_mae[kernel_model], 0.25, axis=0)

        # Create the x-axis data (one point per parameter set)
        x = np.arange(len(parameters[kernel_model]))

        # Create the figure
        fig = go.Figure()

        # Add the line plot for the mean
        fig.add_trace(
            go.Scatter(
                x=x,
                y=avg_mae,
                mode="lines+markers",
                name="Mean MAE",
                text=[
                    str(params) for params in parameters[kernel_model]
                ],  # Add parameter info as hover text
                hoverinfo="text+y",
            )
        )

        # Add the line plot for the q05
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q25,
                mode="lines+markers",
                name="25 perc",
                text=[
                    str(params) for params in parameters[kernel_model]
                ],  # Add parameter info as hover text
                hoverinfo="text+y",
            )
        )

        # Add the line plot for the q95
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q75,
                mode="lines+markers",
                name="75 perc",
                text=[
                    str(params) for params in parameters[kernel_model]
                ],  # Add parameter info as hover text
                hoverinfo="text+y",
            )
        )

        # Add the line plot for the q05
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q05,
                mode="lines+markers",
                name="5 perc",
                text=[
                    str(params) for params in parameters[kernel_model]
                ],  # Add parameter info as hover text
                hoverinfo="text+y",
            )
        )

        # Add the line plot for the q95
        fig.add_trace(
            go.Scatter(
                x=x,
                y=q95,
                mode="lines+markers",
                name="95 perc",
                text=[
                    str(params) for params in parameters[kernel_model]
                ],  # Add parameter info as hover text
                hoverinfo="text+y",
            )
        )

        # Customize layout
        fig.update_layout(
            title="CV results analysis",
            xaxis_title="Parameter Set Index",
            yaxis_title="Value",
            template="plotly_white",
        )

        # Show the plot
        fig.write_html(
            os.path.join(
                "CV_RESULTS",
                "PLOTS",
                f"{model}_{sample_length}_{params_index}_{kernel_model}_2D_plot.html",
            )
        )

        # 3D plots
        # Extract unique C and epsilon pairs
        if kernel_model == "SVR":
            unique_pairs = list(
                {(p["C"], p["epsilon"]) for p in parameters[kernel_model]}
            )

            # Create 3D plots for each unique (C, epsilon) pair
            for C, epsilon in unique_pairs:
                # Filter data for the current pair
                filtered_data = [
                    p
                    for p in parameters[kernel_model]
                    if p["C"] == C and p["epsilon"] == epsilon
                ]
                indices = [
                    i
                    for i, p in enumerate(parameters[kernel_model])
                    if p["C"] == C and p["epsilon"] == epsilon
                ]

                if not filtered_data:
                    continue

                # Extract q_kernel and q_data for the current pair
                q_kernel = np.array([p["q_kernel"] for p in filtered_data])
                q_data = np.array([p["q_data"] for p in filtered_data])
                mae_values = np.array(avg_mae)[indices]

                # Create a mesh grid for q_kernel and q_data
                q_kernel_unique = np.unique(q_kernel)
                q_data_unique = np.unique(q_data)
                q_kernel_grid, q_data_grid = np.meshgrid(q_kernel_unique, q_data_unique)
                mae_grid = np.full(q_kernel_grid.shape, np.nan)

                # Populate mae_grid with corresponding MAE values
                for i, (qk, qd, mae) in enumerate(zip(q_kernel, q_data, mae_values)):
                    grid_idx = np.where((q_kernel_grid == qk) & (q_data_grid == qd))
                    if grid_idx[0].size > 0 and grid_idx[1].size > 0:
                        mae_grid[grid_idx] = mae

                # Create the figure
                fig = go.Figure()

                # Add the 3D surface plot
                fig.add_trace(
                    go.Surface(
                        x=q_kernel_grid,
                        y=q_data_grid,
                        z=mae_grid,
                        colorscale="Viridis",
                        name=f"C={C}, epsilon={epsilon}",
                    )
                )

                # Customize layout
                fig.update_layout(
                    title=f"3D Plot of MAE vs Parameters (C={C}, epsilon={epsilon})",
                    scene=dict(
                        xaxis_title="q_kernel", yaxis_title="q_data", zaxis_title="MAE"
                    ),
                    template="plotly_white",
                )

                fig.write_html(
                    os.path.join(
                        "CV_RESULTS",
                        "PLOTS",
                        f"{model}_{sample_length}_{params_index}_{kernel_model}_{round(C, 3)}_{round(epsilon, 3)}_3D_plot.html",
                    )
                )

        elif kernel_model == "cSVR":
            # Extract values from parameter_data
            q_data_naive = np.array(
                [p["q_data_naive"] for p in parameters[kernel_model]]
            )
            q_kernel_naive = np.array(
                [p["q_kernel_naive"] for p in parameters[kernel_model]]
            )
            mae_values = np.array(avg_mae)

            # Create a mesh grid for q_data_naive and q_kernel_naive
            q_data_unique = np.unique(q_data_naive)
            q_kernel_unique = np.unique(q_kernel_naive)
            q_data_grid, q_kernel_grid = np.meshgrid(q_data_unique, q_kernel_unique)
            mae_grid = np.full(q_data_grid.shape, np.nan)

            # Populate mae_grid with corresponding MAE values
            for qd, qk, mae in zip(q_data_naive, q_kernel_naive, mae_values):
                grid_idx = np.where((q_data_grid == qd) & (q_kernel_grid == qk))
                if grid_idx[0].size > 0 and grid_idx[1].size > 0:
                    mae_grid[grid_idx] = mae

            # Create the 3D plot
            fig = go.Figure()

            # Add the 3D surface plot
            fig.add_trace(
                go.Surface(
                    x=q_kernel_grid, y=q_data_grid, z=mae_grid, colorscale="Viridis"
                )
            )

            # Customize layout
            fig.update_layout(
                title="3D Plot of MAE vs q_data_naive and q_kernel_naive",
                scene=dict(
                    xaxis_title="q_kernel_naive",
                    yaxis_title="q_data_naive",
                    zaxis_title="MAE",
                ),
                template="plotly_white",
            )

            # Show the figure
            fig.write_html(
                os.path.join(
                    "CV_RESULTS",
                    "PLOTS",
                    f"{model}_{sample_length}_{params_index}_{kernel_model}_3D_plot.html",
                )
            )

        elif kernel_model == "ccSVR":
            # Extract unique div_impact values
            unique_div_impact = np.unique(
                [p["div_impact"] for p in parameters[kernel_model]]
            )

            # Create 3D plots for each unique div_impact value
            for div_impact in unique_div_impact:
                # Filter data for the current div_impact value
                filtered_data = [
                    p for p in parameters[kernel_model] if p["div_impact"] == div_impact
                ]
                indices = [
                    i
                    for i, p in enumerate(parameters[kernel_model])
                    if p["div_impact"] == div_impact
                ]

                if not filtered_data:
                    continue

                # Extract q_kernel_div and q_data_div for the current div_impact
                q_kernel_div = np.array([p["q_kernel_div"] for p in filtered_data])
                q_data_div = np.array([p["q_data_div"] for p in filtered_data])
                mae_values = np.array(avg_mae)[indices]

                # Create a mesh grid for q_kernel_div and q_data_div
                q_kernel_unique = np.unique(q_kernel_div)
                q_data_unique = np.unique(q_data_div)
                q_kernel_grid, q_data_grid = np.meshgrid(q_kernel_unique, q_data_unique)
                mae_grid = np.full(q_kernel_grid.shape, np.nan)

                # Populate mae_grid with corresponding MAE values
                for i, (qk, qd, mae) in enumerate(
                    zip(q_kernel_div, q_data_div, mae_values)
                ):
                    grid_idx = np.where((q_kernel_grid == qk) & (q_data_grid == qd))
                    if grid_idx[0].size > 0 and grid_idx[1].size > 0:
                        mae_grid[grid_idx] = mae

                # Create the figure
                fig = go.Figure()

                # Add the 3D surface plot
                fig.add_trace(
                    go.Surface(
                        x=q_kernel_grid,
                        y=q_data_grid,
                        z=mae_grid,
                        colorscale="Viridis",
                        name=f"div_impact={div_impact}",
                    )
                )

                # Customize layout
                fig.update_layout(
                    title=f"3D Plot of MAE vs Parameters (div_impact={div_impact})",
                    scene=dict(
                        xaxis_title="q_kernel_div",
                        yaxis_title="q_data_div",
                        zaxis_title="MAE",
                    ),
                    template="plotly_white",
                )

                fig.write_html(
                    os.path.join(
                        "CV_RESULTS",
                        "PLOTS",
                        f"{model}_{sample_length}_{params_index}_{kernel_model}_3D_plot_div_impact_{round(div_impact, 3)}.html",
                    )
                )
