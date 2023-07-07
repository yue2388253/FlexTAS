import json
import networkx as nx
from src.network.net import Flow


def _generate_net_from_json(json_topo) -> nx.Graph:
    return nx.node_link_graph(json_topo, directed=False, multigraph=False,
                              attrs=dict(source='sourceNodeId', target='destNodeId', name='id',
                                         key='key', link='links')
                              )


def _generate_flows_from_json(json_flows_schedules) -> list[Flow]:
    res = []
    for json_flow_schedule in json_flows_schedules:
        json_flow = json_flow_schedule['flowInfo']
        flow_id = json_flow['id']
        src_id = json_flow['talker']
        assert len(json_flow['listeners']) == 1, "only support uni-cast flows"
        dst_id = json_flow['listeners'][0]

        schedule_node = json_flow_schedule['schedulePath']['scheduleNode']
        path = [(node_pair['preNodeId'], node_pair['currentNodeId']) for node_pair in schedule_node]
        # the first node is `(null, src_id)`, which is unused. drop it.
        path = path[1:]

        for i in range(len(path) - 1):
            assert path[i][1] == path[i+1][0], f"invalid path {path}"

        period = int(json_flow['period'])
        payload = int(json_flow['maxFrameSize'])
        e2e_delay = int(json_flow['ddl'])
        jitter = int(json_flow['jitter'])
        res.append(Flow(flow_id, src_id, dst_id, path, period, payload, e2e_delay, jitter))
    return res


def generate_net_flows_from_json(filename) -> (nx.Graph, list[Flow]):
    with open(filename, 'r') as f:
        json_input = json.load(f)

    json_topo = json_input['input']['topology']
    net = _generate_net_from_json(json_topo)
    net = nx.DiGraph(net)

    json_flows_schedules = json_input['data']['flowsSchedules']
    flows = _generate_flows_from_json(json_flows_schedules)

    return net, flows
