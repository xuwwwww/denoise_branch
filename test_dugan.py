import argparse
from models import model_dict, TrainTask

if __name__ == '__main__':
    # reference https://stackoverflow.com/questions/38050873/can-two-python-argparse-objects-be-combined/38053253
    default_parser = TrainTask.build_default_options()
    default_opt, unknown_opt = default_parser.parse_known_args()
    MODEL = model_dict[default_opt.model_name]
    private_parser = MODEL.build_options()
    opt = private_parser.parse_args(unknown_opt, namespace=default_opt)

    model = MODEL(opt)
    
    if opt.resume_iter > 0:
        print(f"Loading checkpoint from iteration {opt.resume_iter}...")
        model.logger.load_checkpoints(opt.resume_iter)
    else:
        print("Warning: No checkpoint loaded. Testing with random initialization. Use --resume_iter to specify checkpoint.")

    print("Starting evaluation...")
    model.test(opt.resume_iter)
    print("Evaluation complete.")
