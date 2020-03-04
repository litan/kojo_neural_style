class NeuralStyleFilter(script: String, model: String) extends ImageOp {
    import java.awt.image.BufferedImage
    import jep.SharedInterpreter
    import java.io.File
    var interp: SharedInterpreter = _

    def filter(src: BufferedImage) = {
        try {
            interp = new SharedInterpreter()
            val file = writeImageToFile(src)
            fixDynloadPath()
            val styledFile = runStyleTransfer(file)
            image(styledFile)
        }
        catch {
            case t: Throwable =>
                println(t.getMessage)
                src
        }
        finally {
            if (interp != null) {
                interp.close()
            }
        }
    }

    def writeImageToFile(src: BufferedImage): String = {
        import javax.imageio.ImageIO
        val inFile = File.createTempFile("neural-style-in", ".png")
        inFile.deleteOnExit()
        ImageIO.write(removeAlphaChannel(src), "png", inFile)
        inFile.getAbsolutePath
    }

    def runStyleTransfer(file: String): String = {
        val outFile = File.createTempFile("neural-style-out", ".png")
        outFile.deleteOnExit()
        interp.runScript(script)
        interp.exec(s"""
stylize(
    content_image="$file",
    content_scale=1,
    model="$model",
    output_image="$outFile",
    cuda=0
)
    """)
        outFile.getAbsolutePath
    }

    def removeAlphaChannel(img: BufferedImage, color: Color = white): BufferedImage = {
        if (!img.getColorModel().hasAlpha()) {
            return img
        }

        val target = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB)
        val g = target.createGraphics()
        g.setColor(color)
        g.fillRect(0, 0, img.getWidth(), img.getHeight())
        g.drawImage(img, 0, 0, null)
        g.dispose()
        target;
    }

    def fixDynloadPath() {
        interp.exec("""
import sys
p_path = list(filter(lambda n: n.endswith("python3.8"), sys.path))
if len(p_path) == 1:
    sys.path = list(filter(lambda n: not n.endswith("lib-dynload"), sys.path))
    dload = p_path[0] + "/lib-dynload"
    sys.path.append(dload) 
"""
        )
    }
}
