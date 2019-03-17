"""
    Copyright 2019 Javier Lancha Vázquez

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import sys, os

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage:", ". <steps> <snap_step> <load_weights=0/1> <save_weights=0/1>")
        exit(0)

    import keras
    from gan_models import MNIST_GAN

    steps = (int)(sys.argv[1])
    snap_step = (int)(sys.argv[2])
    load_w = (int)(sys.argv[3]) != 0
    save_w = (int)(sys.argv[4]) != 0

    gen_dir = "gen"
    if not os.path.exists(gen_dir):
        os.makedirs(gen_dir)

    GAN = MNIST_GAN(load_weights=load_w)
    print("Train for %d epochs" % steps)
    GAN.train(steps, save_weights=save_w, snap_step=snap_step)
