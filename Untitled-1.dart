{
            "role": "user",
            "content": [
                {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/Raw_Step_image_v3/Raw_Step_10.png"},
                {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/Depth_Step_image_v3_generated/Raw_Step_10_depth.png"},
                {"type": "text", "text": "RGB and depth image of another environment."}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Waypoints: [(450, 800), (160, 780), (850, 900)]"}]
        },
                {
            "role": "user",
            "content":[
                {"type": "image", "image": "/mntdata/main/sim_nav/unreal_data/test_pixel/Raw_Step_10.png"},
                {"type": "text", "text": "The waypoints are projected on the image, all the waypoints are on ground. they are not on any objects or wall"}
            ]
        },