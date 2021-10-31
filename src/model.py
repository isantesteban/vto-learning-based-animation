import os
import numpy as np
import tensorflow.compat.v1 as tf

from . import skinning
from . import smpl
from . import postprocess


SMPL_PATH = "assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
GARMENT_FIT_NET_PATH = "trained_models/tshirt/garment_fit"
GARMENT_WRINKLE_NET_PATH = "trained_models/tshirt/garment_wrinkles"
RECURRENT_STATE_SIZE = 1500


def run_model(motion):
    body = smpl.SMPLModel(SMPL_PATH)

    with tf.Session() as sess:
        import_model(GARMENT_FIT_NET_PATH, sess, scope="fit_regressor")
        import_model(GARMENT_WRINKLE_NET_PATH, sess, scope="wrinkle_regressor")

        v_garment, v_body = compute_garment_deformation(sess, motion, body)
 
    return v_garment, v_body


def import_model(model_dir, sess, scope):
    graph_path = os.path.join(model_dir, 'model.meta')
    saver = tf.train.import_meta_graph(graph_path, import_scope=scope)

    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint_path)


def compute_garment_deformation(sess, motion, body):
    num_frames = len(motion['pose'])
    thetas = motion['pose']
    trans = motion['translation']
    beta = motion['shape']
    betas = np.tile(beta, [num_frames, 1])

    # The model is only trained for 4 shape coefficients,
    # set the others to zero
    betas_limited = betas.copy()
    betas_limited[:, 4:] = 0 

    # Compute skinning weights
    nn = np.loadtxt("assets/meshes/tshirt_closest_body_vertices.txt", delimiter=", ", dtype=int)
    skinning_weights = body.weights[nn]
    num_vertices = skinning_weights.shape[0]

    # Run garment fit regressor
    v_fit = sess.run("fit_regressor/predict:0", {"fit_regressor/x:0": betas_limited[0:1]})
    v_fit = v_fit.reshape((num_vertices, 3))

    # Run recurrent regressor frame by frame
    x = np.concatenate((betas_limited, thetas[:, 3:]), axis=1)
    initial_state = np.zeros((1, RECURRENT_STATE_SIZE))

    v_garment = []
    v_body = []
    for frame in range(num_frames):
        print(f"[INFO] Computing frame {frame+1} of {num_frames}")

        body.set_params(beta=betas[frame], pose=thetas[frame], trans=trans[frame])

        displacements, initial_state = sess.run(["wrinkle_regressor/predict:0", "wrinkle_regressor/rnn/state:0"], {
            "wrinkle_regressor/x:0": [x[frame:frame+1]],
            "wrinkle_regressor/rnn/initial_state:0": initial_state
        })

        displacements = displacements.reshape((num_vertices, 3))
        v_unposed = v_fit + displacements + body.pose_blendshape[nn]
        v_skinned = skinning.lbs(v_unposed, body.global_joint_transforms, skinning_weights) + trans[frame]
        v_skinned = postprocess.fix_collisions(v_skinned, body.verts, body.faces)

        v_garment.append(v_skinned)
        v_body.append(body.verts)

    return np.array(v_garment), np.array(v_body)
