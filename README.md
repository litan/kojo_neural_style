This repo contains code and instructions to support Neural Style Transfer in Kojo.

Neural Style transfer in Kojo is based on [PyTorch](https://pytorch.org/) and builds upon the [fast neural style](https://github.com/pytorch/examples/tree/master/fast_neural_style) example in the PyTorch examples repo.

## Instructions to get going (WIP)
* Install PyTorch using [miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Install [JEP](https://github.com/ninia/jep)
* Copy libk/jep-3.9.0.jar from this repo into ~/.kojo/lite/libk
* Copy include/neural-style.kojo from this repo into ~/kojo-includes (or wherever)
* Copy the style_transfer dir from this repo into ~/work (or wherever)
* Add the following line (suitably adapted) to ~/.kojo/lite/kojo.properties  
`library.path=/home/lalit/miniconda3/envs/pytorch/lib:/home/lalit/miniconda3/envs/pytorch/lib/python3.8/site-packages/jep`

At this point, you are good to go ;).

Here's an example:
```scala
// #include ~/kojo-includes/neural-style
cleari()

val fltr1 = new NeuralStyleFilter(
    "/home/lalit/work/kojo_neural_style/neural_style/run.py",
    "/home/lalit/work/kojo_neural_style/neural_style/saved_models/mosaic.pth"
)

val fltr2 = new NeuralStyleFilter(
    "/home/lalit/work/kojo_neural_style/neural_style/run.py",
    "/home/lalit/work/kojo_neural_style/neural_style/saved_models/udnie.pth"
)

val drawing = Picture {
    setPenColor(black)
    setPenThickness(10)
    repeat(18) {
        repeat(5) {
            forward(100)
            right(72)
        }
        right(20)
    }
}

val pic = effect(fltr2) * effect(fltr1) -> drawing
draw(pic)
```
![example1](./doc/example1.png)

