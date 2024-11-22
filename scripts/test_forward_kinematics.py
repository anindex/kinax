import time

from brax import test_utils
from brax import kinematics



# def load(path: Union[str, epath.Path]):
#     """Loads a brax system from a MuJoCo mjcf file path."""
#     elem = ElementTree.fromstring(epath.Path(path).read_text())
#     # _fuse_bodies(elem)
#     meshdir = _get_meshdir(elem)
#     assets = _find_assets(elem, epath.Path(path), meshdir)
#     xml = ElementTree.tostring(elem, encoding='unicode')
#     mj = mujoco.MjModel.from_xml_string(xml, assets=assets)

#     return load_model(mj)


# def load_fixture(path: str) -> System:
#     full_path = epath.resource_path('brax') / f'test_data/{path}'
#     if not full_path.exists():
#         full_path = epath.resource_path('brax') / f'envs/assets/{path}'
#     sys = load(full_path)

#     return sys


if __name__ == "__main__":
    xml_file = "ant.xml"
    sys = test_utils.load_fixture(xml_file)
    print(sys.num_links(), sys.link_names, sys.link_types)
    print(sys.link_parents)

    for mj_prev, mj_next in test_utils.sample_mujoco_states(
            xml_file, count=50, random_init=True, vel_to_local=False):
        
        tic = time.time()
        x, xd = kinematics.forward(sys, mj_prev.qpos, mj_prev.qvel)
        print(mj_prev.qpos.shape, mj_prev.qvel.shape)
        print(x.pos.shape, xd.ang.shape)
        toc = time.time()
        print("jax forward kinematics time: ", toc-tic)
