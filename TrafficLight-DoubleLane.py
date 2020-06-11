"""Grid/green wave example."""

import json

from flow.utils.registry import make_create_env
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from flow.core.experiment import Experiment
from grid2 import SimpleGridScenario2
from DoubleLaneNetwork import DoubleLaneNetwork
from flow.envs.traffic_light_grid import TrafficLightGridPOEnv
from flow.core.params import TrafficLightParams
from flow.core.params import SumoLaneChangeParams
from flow.networks import BayBridgeNetwork
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter
from flow.controllers.routing_controllers import MinicityRouter


# time horizon of a single rollout
HORIZON = 2000
# number of rollouts per training iteration
#N_ROLLOUTS = 20
N_ROLLOUTS = 1
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


def get_flow_params(col_num, row_num, additional_net_params):
    initial_config = InitialConfig(
        spacing='custom', lanes_distribution=float('inf'), shuffle=True)

    inflow = InFlows()
    outer_edges = gen_edges(col_num, row_num)
    for i in range(len(outer_edges)):
        inflow.add(
            veh_type='idm',
            edge=outer_edges[i],
            probability=0.05,
            departLane='free',
            departSpeed=20)

    net_params = NetParams(
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
        'switch_time': 3.0,
        ##'num_observed': 2,
        'num_observed': 10,
        'discrete': False,
        'tl_type': 'controlled' ## If controlled, then add "traffic_lights": true in additional_net_params
        #'tl_type': 'actuated' ## If actuated, remove "traffic_lights": true in additional_net_params
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
# vehicles.add(
#     veh_id='idm',
#     acceleration_controller=(SimCarFollowingController, {}),
#     car_following_params=SumoCarFollowingParams(
#         minGap=2.5,
#         max_speed=V_ENTER,
#         speed_mode="all_checks",
#     ),
#     routing_controller=(GridRouter, {}),
#     num_vehicles=tot_cars)

initial_config, net_params = \
    get_flow_params(N_ROWS,N_COLUMNS, additional_net_params)

# initial_config, net_params = \
#     get_non_flow_params(V_ENTER, additional_net_params)

# traffic_lights = TrafficLightParams(baseline=False)
# phases = [{
#             "duration": "38",
#             "minDur": "8",
#             "maxDur": "45",
#             "state": "GGGrrrGGGrrr"
#         }, {
#             "duration": "7",
#             "minDur": "3",
#             "maxDur": "7",
#             "state": "yyyrrryyyrrr"
#         }, {
#             "duration": "38",
#             "minDur": "3",
#             "maxDur": "45",
#             "state": "rrrGGGrrrGGG"
#         }, {
#             "duration": "7",
#             "minDur": "3",
#             "maxDur": "7",
#             "state": "rrryyyrrryyy"
#         }]


# traffic_lights.add(node_id="center0", phases=phases)
# #print("center"+str(i))

        
        

flow_params = dict(
    # name of the experiment
    exp_tag='green_wave',

    # name of the flow environment the experiment is running on
    env_name=TrafficLightGridPOEnv,

    # name of the scenario class the experiment is running on
    network=DoubleLaneNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        #render=False,
        render=True,
        restart_instance=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=net_params,

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=initial_config,

    #tls = traffic_lights
)

# exp = Experiment(flow_params)
# exp.run(1, convert_to_csv=False)