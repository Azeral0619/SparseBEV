_base_ = ["./r50_nuimg_704x256_400q_36ep.py"]

test_pipeline = [dict(type="LoadImageToBase64", num_views=6)]
data = dict(val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
