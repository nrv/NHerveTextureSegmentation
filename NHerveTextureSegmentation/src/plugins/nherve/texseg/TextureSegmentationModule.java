package plugins.nherve.texseg;

import javax.swing.JPanel;

import plugins.nherve.toolbox.imageanalysis.ImageAnalysisContext;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisException;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisParameters;
import plugins.nherve.toolbox.imageanalysis.impl.WithGUIModuleDefaultImpl;

public class TextureSegmentationModule extends WithGUIModuleDefaultImpl {
	private final static String PLUGIN_NAME = "Texture segmentation";
	
	public TextureSegmentationModule() {
		super(PLUGIN_NAME);
	}

	@Override
	public void populateGUI(ImageAnalysisParameters defaultParameters, JPanel panel) {
	}

	@Override
	public boolean analyze(ImageAnalysisContext context) throws ImageAnalysisException {
		return false;
	}

	@Override
	public void getParametersFromGui(JPanel p, ImageAnalysisContext c) {
	}

	@Override
	public void populateWithDefaultParameterValues(ImageAnalysisParameters parameters) {
	}

}
