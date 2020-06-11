"""Grid/green wave example."""

import json

from flow.utils.registry import make_create_env
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.experiment import Experiment
from StaticAgent import static_rl_actions
from flow.utils.registry import make_create_env

##########CONFIGURATION
isTesting = False

#####


# time horizon of a single rollout
HORIZON = 200
# number of rollouts per training iteration
N_ROLLOUTS = 30
# number of parallel workers
#N_CPUS = 2
N_CPUS = 1


def gen_edges(row_num, col_num):
    edges = []
    for i in range(col_num):
        edges += ['left' + str(row_num) + '_' + str(i)]
        edges += ['right' + '0' + '_' + str(i)]

    # build the left and then the right edges
    for i in range(row_num):
        edges += ['bot' + str(i) + '_' + '0']
        edges += ['top' + str(i) + '_' + str(col_num)]

    return edges
def get_inflow_params(col_num, row_num, additional_net_params, inflow_probability):
    """Define the network and initial params in the presence of inflows.
    Parameters
    ----------
    col_num : int
        number of columns in the traffic light grid
    row_num : int
        number of rows in the traffic light grid
    additional_net_params : dict
        network-specific parameters that are unique to the traffic light grid
 
    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the network
    """
    initial = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=False)
 
    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)

    # def getProb(i):
    #     if i < 2:
    #         return 0.05
    #     return 0.15

def get_flow_params(col_num, row_num, additional_net_params):
    initial_config = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=inflow_probability[i],
            depart_lane='free',
            depart_speed=10)
    net = NetParams(
        inflows=inflow,
        additional_params=additional_net_params)

    return initial_config, net_params


def get_non_flow_params(enter_speed, add_net_params):
    """Define the network and initial params in the absence of inflows.

    Note that when a vehicle leaves a network in this case, it is immediately
    returns to the start of the row/column it was traversing, and in the same
    direction as it was before.

    Parameters
    ----------
    enter_speed : float
        initial speed of vehicles as they enter the network.
    add_net_params: dict
        additional network-specific parameters (unique to the grid)

    Returns
    -------
    flow.core.params.InitialConfig
        parameters specifying the initial configuration of vehicles in the
        network
    flow.core.params.NetParams
        network-specific parameters used to generate the scenario
    """
    additional_init_params = {'enter_speed': enter_speed}
    initial_config = InitialConfig(
        spacing='custom', additional_params=additional_init_params)
    net_params = NetParams(additional_params=add_net_params)

    return initial_config, net_params


V_ENTER = 30
INNER_LENGTH = 300
LONG_LENGTH = 100
SHORT_LENGTH = 300
N_ROWS = 1
N_COLUMNS = 1
# NUM_CARS_LEFT = 1
# NUM_CARS_RIGHT = 1
# NUM_CARS_TOP = 1
# NUM_CARS_BOT = 1
NUM_CARS_LEFT = 10
NUM_CARS_RIGHT = 10
NUM_CARS_TOP = 10
NUM_CARS_BOT = 10
tot_cars = (NUM_CARS_LEFT + NUM_CARS_RIGHT) * N_COLUMNS \
           + (NUM_CARS_BOT + NUM_CARS_TOP) * N_ROWS

grid_array = {
    "short_length": SHORT_LENGTH,
    "inner_length": INNER_LENGTH,
    "long_length": LONG_LENGTH,
    "row_num": N_ROWS,
    "col_num": N_COLUMNS,
    "cars_left": NUM_CARS_LEFT,
    "cars_right": NUM_CARS_RIGHT,
    "cars_top": NUM_CARS_TOP,
    "cars_bot": NUM_CARS_BOT
}

additional_env_params = {
        'target_velocity': 50,
        'switch_time': 4.0,
        'num_observed': 2,
        'discrete': True,
        'tl_type': 'controlled'
    }


additional_net_params = {
    'speed_limit': 35,
    'grid_array': grid_array,
    'horizontal_lanes': 2,
    'vertical_lanes': 2,
    'traffic_lights': True ######### !
}

vehicles = VehicleParams()
vehicles.add( 
    veh_id='idm', 
    acceleration_controller=(SimCarFollowingController, {}),
    car_following_params=SumoCarFollowingParams( min_gap=2.5, max_speed=V_ENTER, speed_mode="all_checks",tau=1.1 ), 
    lane_change_params=SumoLaneChangeParams( lane_change_mode="strategic", model="LC2013", ), 
    lane_change_controller=(StaticLaneChanger, {}), 
    routing_controller=(MinicityRouter, {}),
    num_vehicles=tot_cars)
 
# collect the initialization and network-specific parameters based on the
# choice to use inflows or not
# if USE_INFLOWS:
#     initial_config, net_params = get_inflow_params(
#         col_num=N_COLUMNS,
#         row_num=N_ROWS,
#         additional_net_params=additional_net_params)
# else:
#     initial_config, net_params = get_non_flow_params(
#         enter_speed=V_ENTER,
#         add_net_params=additional_net_params)
 



t = 0

def rl_actions(state):
    global t
    t += 1
    return [t%30 == 0]    


def GetTrafficLightEnv(inflow_probability, render=False, evaluate=False):
    initial_config, net_params = get_inflow_params(
        col_num=N_COLUMNS,
        row_num=N_ROWS,
        additional_net_params=additional_net_params,
        inflow_probability=inflow_probability)
    
    flow_params = dict(
        # name of the experiment
        exp_tag='traffic_light_grid',
    
        # name of the flow environment the experiment is running on
        env_name=TrafficLightGridPOEnv,
    
        # name of the network class the experiment is running on
        network=TrafficLightGridNetwork,
    
        # simulator that is used by the experiment
        simulator='traci',
        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=1,
            render=render,
            emission_path="Results",
            restart_instance=True
        ),
    
        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params=additional_env_params,
            evaluate=evaluate
        ),
    
        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component). This is
        # filled in by the setup_exps method below.
        net=net_params,
    
        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,
    
        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig). This is filled in by the
        # setup_exps method below.
        initial=initial_config,
    )
    
    # Get the env name and a creator for the environment.
    create_env, _ = make_create_env(flow_params)
    # Create the environment.
    env = create_env()
    return env

# def getFlowParamsForTls():
#     return flow_params

# exp = Experiment(flow_params)
# exp.run(2, convert_to_csv=False)
