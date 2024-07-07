import matplotlib.pyplot as plt
import io
import base64
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import json
import dash

def create_interactive_visualization(env):
    app = Dash(__name__)
    state_action_log = []

    def generate_figure():
        #fig = env.visualize_2d_world(return_fig=True)
        #buf = io.BytesIO()
        env
        fig.savefig(buf, format="png")
        plt.close(fig)
        data = base64.b64encode(buf.getbuffer()).decode("utf8")
        buf.close()
        return data

    app.layout = html.Div([
        html.Img(id='network-graph', src='data:image/png;base64,{}'.format(generate_figure())),
        html.Div(id='button-bar', children=[
            html.Button(location, id={'type': 'location-button', 'index': location})
            for location in env.data_centers
        ], style={'display': 'flex', 'justify-content': 'center', 'flexWrap': 'wrap'}),
        dcc.Store(id='state-action-log', data=[]),
        html.Div(id='log-output')
    ])

    @app.callback(
        [Output('state-action-log', 'data'), Output('log-output', 'children'), Output('network-graph', 'src')],
        [Input({'type': 'location-button', 'index': dash.dependencies.ALL}, 'n_clicks')],
        [State('state-action-log', 'data')]
    )
    def update_log(n_clicks, current_log):
        ctx = dash.callback_context
        if not ctx.triggered:
            return current_log, '', 'data:image/png;base64,{}'.format(generate_figure())

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        location = json.loads(button_id)['index']

        # Log the action and update the environment state
        action = (location, location)  # For demonstration, replace with actual action logic
        state, reward, done = env.step(action)
        current_log.append({'state': state, 'action': action, 'reward': reward})

        return current_log, json.dumps(current_log, indent=2), 'data:image/png;base64,{}'.format(generate_figure())

    app.run_server(debug=True)