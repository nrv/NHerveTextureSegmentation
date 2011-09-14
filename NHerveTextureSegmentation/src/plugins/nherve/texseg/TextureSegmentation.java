package plugins.nherve.texseg;

import icy.gui.component.ComponentUtil;
import icy.gui.util.GuiUtil;
import icy.image.IcyBufferedImage;
import icy.preferences.XMLPreferences;
import icy.system.thread.ThreadUtil;
import icy.type.TypeUtil;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.swing.Box;
import javax.swing.ButtonGroup;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JRadioButton;
import javax.swing.JTextField;
import javax.swing.border.TitledBorder;

import plugins.nherve.maskeditor.MaskEditor;
import plugins.nherve.toolbox.NherveToolbox;
import plugins.nherve.toolbox.concurrent.MultipleDataTask;
import plugins.nherve.toolbox.concurrent.TaskException;
import plugins.nherve.toolbox.concurrent.TaskManager;
import plugins.nherve.toolbox.image.BinaryIcyBufferedImage;
import plugins.nherve.toolbox.image.feature.descriptor.MultiThreadedSignatureExtractor;
import plugins.nherve.toolbox.image.feature.learning.SVMClassifier;
import plugins.nherve.toolbox.image.feature.signature.VectorSignature;
import plugins.nherve.toolbox.image.mask.Mask;
import plugins.nherve.toolbox.image.mask.MaskException;
import plugins.nherve.toolbox.image.mask.MaskStack;
import plugins.nherve.toolbox.image.toolboxes.ColorSpaceTools;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisContext;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisException;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisModule;
import plugins.nherve.toolbox.imageanalysis.ImageAnalysisProcessListener;
import plugins.nherve.toolbox.imageanalysis.modules.LBPSignaturesExtractionModule;
import plugins.nherve.toolbox.imageanalysis.modules.PixelSignatureData;
import plugins.nherve.toolbox.imageanalysis.modules.SquareRegionsExtractionModule;
import plugins.nherve.toolbox.libsvm.svm;
import plugins.nherve.toolbox.libsvm.svm_parameter;
import plugins.nherve.toolbox.plugin.HelpWindow;
import plugins.nherve.toolbox.plugin.SingletonPlugin;

public class TextureSegmentation extends SingletonPlugin implements ActionListener {
	public final static String CD_SVM = "SVM";
	public final static String CD_PREDICTION = "PREDICTION";
	public final static String CD_WIDTH = "WIDTH";

	private class SVMWorker implements Runnable {
		public class SVMPredictionWorker extends MultipleDataTask<PixelSignatureData, Integer> {
			private double[] predictionData;
			private SVMClassifier svm;
			private int w;

			public SVMPredictionWorker(List<PixelSignatureData> allData, int idx1, int idx2) {
				super(allData, idx1, idx2);

				predictionData = null;
				svm = null;
				w = 0;
			}

			@Override
			public void call(PixelSignatureData data, int idx) throws Exception {
				predictionData[(int) data.pix.x + (int) data.pix.y * w] = svm.rawScore(data.sig);
				done++;

				if (done >= donePct) {
					while (done >= donePct) {
						donePct += stepPct;
					}

					ThreadUtil.invokeLater(new Runnable() {
						@Override
						public void run() {
							pbSVM.setValue(done);
						}
					});
				}
			}

			@Override
			public Integer outputCall() throws Exception {
				return 0;
			}

			@Override
			public void processContextualData() {
				predictionData = (double[]) getContextualData(CD_PREDICTION);
				svm = (SVMClassifier) getContextualData(CD_SVM);
				w = (Integer) getContextualData(CD_WIDTH);
			}

		}

		public SVMWorker(ImageAnalysisContext context) {
			super();
			this.context = context;
		}

		private ImageAnalysisContext context;

		@Override
		public void run() {
			MaskStack stack = context.getStack();

			Mask pm = stack.getByLabel(POSITIVE_MASK);
			if (pm == null) {
				logError("Unable to find mask " + POSITIVE_MASK);
				return;
			}

			Mask nm = stack.getByLabel(NEGATIVE_MASK);
			if (nm == null) {
				logError("Unable to find mask " + NEGATIVE_MASK);
				return;
			}

			ThreadUtil.invokeLater(new Runnable() {
				@Override
				public void run() {
					pbSVM.setStringPainted(true);
					pbSVM.setIndeterminate(false);
					pbSVM.setString("Learning ...");
				}
			});

			@SuppressWarnings("unchecked")
			final List<PixelSignatureData> myData = (List<PixelSignatureData>) context.getObject(sigs);

			List<VectorSignature> posData = new ArrayList<VectorSignature>();
			for (PixelSignatureData aData : myData) {
				if (pm.contains(aData.pix)) {
					posData.add(aData.sig);
				}
			}

			List<VectorSignature> negData = new ArrayList<VectorSignature>();
			for (PixelSignatureData aData : myData) {
				if (nm.contains(aData.pix)) {
					negData.add(aData.sig);
				}
			}

			if (!stopped) {
				if (!posData.isEmpty() && !negData.isEmpty()) {
					try {
						SVMClassifier svm = new SVMClassifier();
						svm.createProblem(posData, negData);
						svm.setC(Double.parseDouble(prmC.getText()));
						svm.setGamma(Double.parseDouble(prmG.getText()));

						if (rbKernelLin.isSelected()) {
							svm.setKernel(svm_parameter.LINEAR);
						} else if (rbKernelTri.isSelected()) {
							svm.setKernel(svm_parameter.TRIANGULAR);
						} else if (rbKernelRBF.isSelected()) {
							svm.setKernel(svm_parameter.RBF);
						}

						svm.learnModel();

						ThreadUtil.invokeLater(new Runnable() {
							@Override
							public void run() {
								pbSVM.setString(null);
								pbSVM.setMaximum(myData.size());
								pbSVM.setValue(0);
								pbTexture.setIndeterminate(false);
							}
						});

						done = 0;
						stepPct = myData.size() / 100;
						if (stepPct == 0) {
							stepPct = 1;
						}
						donePct = stepPct;

						IcyBufferedImage img = context.getWorkingImage();
						int w = img.getWidth();
						int h = img.getHeight();

						IcyBufferedImage svmpred = new IcyBufferedImage(w, h, 1, TypeUtil.TYPE_DOUBLE);
						double[] predictionData = svmpred.getDataXYAsDouble(0);
						Arrays.fill(predictionData, 0d);

						if (!stopped) {

							TaskManager tm = TaskManager.getMainInstance();
							try {
								Map<String, Object> contextualData = new HashMap<String, Object>();
								contextualData.put(CD_PREDICTION, predictionData);
								contextualData.put(CD_SVM, svm);
								contextualData.put(CD_WIDTH, w);
								tm.submitMultiForAll(myData, contextualData, SVMPredictionWorker.class, this, "", 0);
							} catch (TaskException e) {
								e.printStackTrace();
							} catch (InterruptedException e) {
								e.printStackTrace();
							}
							svmpred.dataChanged();

							ThreadUtil.invokeLater(new Runnable() {
								@Override
								public void run() {
									pbSVM.setString("Done");
									pbSVM.setValue(myData.size());
								}
							});

							stack.beginUpdate();
							Mask am = stack.getActiveMask();
							Mask svmMsk = stack.getByLabel(PREDICTION_MASK);
							if (svmMsk != null) {
								stack.remove(svmMsk);
							}

							svmMsk = stack.createNewMask(PREDICTION_MASK, false, Color.BLUE, MaskEditor.getRunningInstance(false).getGlobalOpacity());
							byte[] bb = svmMsk.getBinaryData().getRawData();
							for (int i = 0; i < bb.length; i++) {
								bb[i] = (predictionData[i] > 0 ? BinaryIcyBufferedImage.TRUE : BinaryIcyBufferedImage.FALSE);
							}
							svmMsk.getBinaryData().dataChanged();
							stack.setActiveMask(am);
							stack.endUpdate();
						}
					} catch (Exception e) {
						e.printStackTrace();
						return;
					}
				} else {
					logError("You should fill positive and negative masks !");
				}
			}

			ThreadUtil.invokeLater(new Runnable() {
				@Override
				public void run() {
					btStop.setEnabled(false);
					btSVM.setEnabled(true);
				}
			});

			svmWorkerThread = null;
		}
	}

	private class TextureWorker implements Runnable, ImageAnalysisProcessListener, MultiThreadedSignatureExtractor.Listener {
		public TextureWorker(ImageAnalysisContext context) {
			super();
			this.context = context;
		}

		private ImageAnalysisContext context;

		@Override
		public void run() {
			ThreadUtil.invokeLater(new Runnable() {
				@Override
				public void run() {
					pbTexture.setStringPainted(true);
					pbTexture.setIndeterminate(false);
					pbTexture.setValue(0);
					pbTexture.setString("Starting ...");
				}
			});

			SquareRegionsExtractionModule rem = new SquareRegionsExtractionModule("Pixels extraction");
			rem.setLogEnabled(cbLog.isSelected());
			rem.populateWithDefaultParameterValues(context);
			rem.setParameter(context, SquareRegionsExtractionModule.PRM_W, prmW.getText());

			LBPSignaturesExtractionModule sigex = new LBPSignaturesExtractionModule("Texture informations extraction");
			sigex.setLogEnabled(cbLog.isSelected());
			sigex.add(this);
			sigex.populateWithDefaultParameterValues(context);
			sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_PIXELS, rem.getParameterInternalName(SquareRegionsExtractionModule.RES_PIXELS));
			sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_REGIONS, rem.getParameterInternalName(SquareRegionsExtractionModule.RES_REGIONS));
			sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_P, prmP.getText());
			sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_R, prmR.getText());
			sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_T, prmT.getText());
			// sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_V,
			// prmV.getText());
			// sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_I,
			// cbIntensity.isSelected());
			if (cbColor.isSelected()) {
				sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_COLOR, ColorSpaceTools.COLOR_SPACES[ColorSpaceTools.RGB]);
			} else {
				sigex.setParameter(context, LBPSignaturesExtractionModule.PRM_COLOR, LBPSignaturesExtractionModule.GRAY);
			}
			sigs = sigex.getParameterInternalName(LBPSignaturesExtractionModule.RES_SIGNATURES);

			try {
				context.processAndWait(rem, true);
				ThreadUtil.invokeLater(new Runnable() {
					@Override
					public void run() {
						pbTexture.setString(null);
					}
				});
				if (!stopped) {
					context.processAndNotify(sigex, this, true);
				}
			} catch (ImageAnalysisException e1) {
				e1.printStackTrace();
			}
		}

		@Override
		public void notifyProcessEnded(ImageAnalysisModule module) {
			ThreadUtil.invokeLater(new Runnable() {
				@Override
				public void run() {
					pbTexture.setStringPainted(true);
					btTexture.setEnabled(true);
					if (stopped) {
						pbTexture.setString("Stopped");
					} else {
						pbTexture.setString("Done");
						btSVM.setEnabled(true);
						btStop.setEnabled(false);
					}
				}
			});

		}

		@Override
		public void notifyProgress(final int nb, final int total) {
			ThreadUtil.invokeLater(new Runnable() {
				@Override
				public void run() {

					pbTexture.setMaximum(total);
					pbTexture.setValue(nb);
				}
			});
		}
	}

	private final static String POSITIVE_MASK = "Positive";
	private final static String NEGATIVE_MASK = "Negative";
	private final static String PREDICTION_MASK = "Prediction";

	private static String HELP = "<html>" + "<p align=\"center\"><b>" + HelpWindow.getTagFullPluginName() + "</b></p>" + "<p align=\"center\"><b>" + NherveToolbox.getDevNameHtml() + "</b></p>" + "<p align=\"center\"><a href=\"http://www.herve.name/pmwiki.php/Main/TextureSegmentation\">Online help is available</a></p>" + "<p align=\"center\"><b>" + NherveToolbox.getCopyrightHtml() + "</b></p>" + "<hr/>" + "<p>" + HelpWindow.getTagPluginName() + NherveToolbox.getLicenceHtml() + "</p>" + "<p>" + NherveToolbox.getLicenceHtmllink() + "</p>" + "</html>";

	private JLabel currentImage;
	private JProgressBar pbTexture;
	private JProgressBar pbSVM;

	private JTextField prmW;
	private JTextField prmP;
	private JTextField prmR;
	private JTextField prmT;
	private JTextField prmC;
	private JTextField prmG;
	// private JTextField prmV;

	private JCheckBox cbColor;
	// private JCheckBox cbIntensity;
	private JCheckBox cbLog;

	private JButton btTexture;
	private JButton btSVM;
	private JButton btStop;
	private JButton btHelp;

	private JRadioButton rbKernelLin;
	private JRadioButton rbKernelRBF;
	private JRadioButton rbKernelTri;

	private boolean stopped;

	private int done;
	private int donePct;
	private int stepPct;
	private ImageAnalysisContext context;
	private String sigs;

	private Thread svmWorkerThread;

	public TextureSegmentation() {
		super();
		stopped = false;
		svmWorkerThread = null;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		setLogEnabled(cbLog.isSelected());

		Object o = e.getSource();

		if (o == null) {
			return;
		}

		if (o instanceof JRadioButton) {

			JRadioButton b = (JRadioButton) e.getSource();
			if (b == rbKernelLin) {
				prmG.setEnabled(false);
			}
			if (b == rbKernelRBF) {
				prmG.setEnabled(true);
			}
			if (b == rbKernelTri) {
				prmG.setEnabled(false);
			}
		}

		if (o instanceof JButton) {
			JButton b = (JButton) e.getSource();
			if (b == null) {
				return;
			}

			if (b == btHelp) {
				openHelpWindow(HELP, 400, 300);
				return;
			}

			if (b == btTexture) {
				if (hasCurrentSequence()) {
					stopped = false;
					btStop.setEnabled(true);
					btTexture.setEnabled(false);
					context = new ImageAnalysisContext();
					context.setWorkingImage(getCurrentSequence().getFirstImage());
					context.setWorkingName(getCurrentSequence().getName());

					MaskEditor me = MaskEditor.getRunningInstance(true);
					MaskStack stack = me.getBackupObject();

					try {
						stack.beginUpdate();
						if (stack.size() == 1) {
							if (stack.getActiveMask().getSurface() == 0) {
								stack.remove(stack.getActiveMask());
							}
						}
						if (stack.getByLabel(NEGATIVE_MASK) == null) {
							stack.createNewMask(NEGATIVE_MASK, false, Color.RED, me.getGlobalOpacity());
						}
						if (stack.getByLabel(POSITIVE_MASK) == null) {
							stack.createNewMask(POSITIVE_MASK, false, Color.GREEN, me.getGlobalOpacity());
						}
						stack.endUpdate();
					} catch (MaskException e1) {
						e1.printStackTrace();
						return;
					}
					context.setStack(stack);

					btSVM.setEnabled(false);

					TextureWorker w = new TextureWorker(context);
					Thread t = new Thread(w);
					t.start();
				}
				return;
			}

			if (b == btSVM) {
				if (hasCurrentSequence()) {
					MaskEditor me = MaskEditor.getRunningInstance(true);
					MaskStack stack = me.getBackupObject();
					if (context != null) {
						context.setStack(stack);
						btStop.setEnabled(true);
						btSVM.setEnabled(false);

						SVMWorker svmWorker = new SVMWorker(context);
						svmWorkerThread = new Thread(svmWorker);
						svmWorkerThread.start();
					}
				}
			}

			if (b == btStop) {
				if ((context != null) || (svmWorkerThread != null)) {
					stopped = true;
					if (context != null) {
						context.stopRunningProcesses();
					}
					if (svmWorkerThread != null) {
						svmWorkerThread.interrupt();
						svmWorkerThread = null;
					}
					btStop.setEnabled(false);
				}
			}
		}
	}

	@Override
	public void sequenceHasChanged() {
		if (hasCurrentSequence()) {
			currentImage.setText(getCurrentSequence().getName());
			btTexture.setEnabled(true);
		} else {
			currentImage.setText("none");
			btTexture.setEnabled(false);
		}

		btSVM.setEnabled(false);
		btStop.setEnabled(false);

		if (context != null) {
			context.stopRunningProcesses();
			context = null;
		}
	}

	@Override
	public void sequenceWillChange() {
	}

	@Override
	public void fillInterface(JPanel mainPanel) {
		XMLPreferences preferences = getPreferences();
		int w = preferences.getInt("w", 5);
		int p = preferences.getInt("p", 8);
		double r = preferences.getDouble("r", 1.);
		double t = preferences.getDouble("t", 25.);
		double c = preferences.getDouble("c", 1.);
		double g = preferences.getDouble("g", 1.);
		// int v = preferences.getInt("v", 1);


		// Current image
		currentImage = new JLabel("none");
		JPanel p1 = GuiUtil.createLineBoxPanel(new JLabel("Current image : "), Box.createHorizontalGlue(), currentImage);
		mainPanel.add(p1);

		mainPanel.add(Box.createHorizontalStrut(25));

		// SLBPriu descriptor
		cbColor = new JCheckBox("Color");
		cbColor.setSelected(false);
		// cbIntensity = new JCheckBox("Intensity");
		// cbIntensity.setSelected(false);
		prmW = new JTextField(Integer.toString(w));
		ComponentUtil.setFixedSize(prmW, new Dimension(60, 25));
		prmP = new JTextField(Integer.toString(p));
		ComponentUtil.setFixedSize(prmP, new Dimension(60, 25));
		prmR = new JTextField(Double.toString(r));
		ComponentUtil.setFixedSize(prmR, new Dimension(60, 25));
		prmT = new JTextField(Double.toString(t));
		ComponentUtil.setFixedSize(prmT, new Dimension(60, 25));
		// prmV = new JTextField(Integer.toString(v));
		// ComponentUtil.setFixedSize(prmV, new Dimension(60, 25));
		// JPanel p3a = GuiUtil.createLineBoxPanel(Box.createHorizontalGlue(),
		// cbColor, Box.createHorizontalGlue(), new JLabel("V "), prmV,
		// Box.createHorizontalGlue(), cbIntensity, Box.createHorizontalGlue());
		// JPanel p3b = GuiUtil.createLineBoxPanel(Box.createHorizontalGlue(),
		// new JLabel("W "), prmW, Box.createHorizontalGlue(), new JLabel("P "),
		// prmP, Box.createHorizontalGlue(), new JLabel("R "), prmR,
		// Box.createHorizontalGlue(), new JLabel("T "), prmT,
		// Box.createHorizontalGlue());

		JPanel p3b = GuiUtil.createLineBoxPanel(Box.createHorizontalGlue(), cbColor, Box.createHorizontalGlue(), new JLabel("W "), prmW, Box.createHorizontalGlue(), new JLabel("P "), prmP, Box.createHorizontalGlue(), new JLabel("R "), prmR, Box.createHorizontalGlue(), new JLabel("T "), prmT, Box.createHorizontalGlue());

		pbTexture = new JProgressBar();
		ComponentUtil.setFixedWidth(pbTexture, 300);
		JPanel p2 = GuiUtil.createLineBoxPanel(new JLabel("Texture : "), Box.createHorizontalGlue(), pbTexture);

		// JPanel p23 = GuiUtil.createPageBoxPanel(Box.createHorizontalGlue(),
		// p3a, Box.createHorizontalGlue(), p3b, Box.createHorizontalGlue(), p2,
		// Box.createHorizontalGlue());
		JPanel p23 = GuiUtil.createPageBoxPanel(Box.createHorizontalGlue(), p3b, Box.createHorizontalGlue(), p2, Box.createHorizontalGlue());
		p23.setBorder(new TitledBorder("SLBPriu descriptor"));
		mainPanel.add(p23);

		mainPanel.add(Box.createHorizontalStrut(25));

		// SVM
		prmC = new JTextField(Double.toString(c));
		ComponentUtil.setFixedSize(prmC, new Dimension(60, 25));
		prmG = new JTextField(Double.toString(g));
		prmG.setEnabled(false);
		ComponentUtil.setFixedSize(prmG, new Dimension(60, 25));

		ButtonGroup kern = new ButtonGroup();
		rbKernelLin = new JRadioButton(svm.kernel_type_table[svm_parameter.LINEAR]);
		rbKernelLin.addActionListener(this);
		kern.add(rbKernelLin);
		rbKernelTri = new JRadioButton(svm.kernel_type_table[svm_parameter.TRIANGULAR]);
		rbKernelTri.addActionListener(this);
		kern.add(rbKernelTri);
		rbKernelRBF = new JRadioButton(svm.kernel_type_table[svm_parameter.RBF]);
		rbKernelRBF.addActionListener(this);
		kern.add(rbKernelRBF);
		rbKernelTri.setSelected(true);
		JPanel p5a = GuiUtil.createLineBoxPanel(Box.createHorizontalGlue(), rbKernelTri, rbKernelLin, rbKernelRBF, Box.createHorizontalGlue(), new JLabel("C "), prmC, Box.createHorizontalGlue(), new JLabel("gamma "), prmG, Box.createHorizontalGlue());
		pbSVM = new JProgressBar();
		ComponentUtil.setFixedWidth(pbSVM, 300);
		JPanel p5b = GuiUtil.createLineBoxPanel(new JLabel("SVM : "), Box.createHorizontalGlue(), pbSVM);
		JPanel p5 = GuiUtil.createPageBoxPanel(Box.createHorizontalGlue(), p5a, Box.createHorizontalGlue(), p5b, Box.createHorizontalGlue());
		p5.setBorder(new TitledBorder("SVM"));
		mainPanel.add(p5);

		mainPanel.add(Box.createHorizontalStrut(25));

		// Buttons
		cbLog = new JCheckBox("Log");
		cbLog.setSelected(false);
		btTexture = new JButton("Texture");
		btTexture.addActionListener(this);
		btSVM = new JButton("SVM");
		btSVM.addActionListener(this);
		btStop = new JButton("Stop");
		btStop.addActionListener(this);
		btHelp = new JButton(NherveToolbox.questionIcon);
		btHelp.addActionListener(this);
		JPanel p4 = GuiUtil.createLineBoxPanel(Box.createHorizontalGlue(), btHelp, Box.createHorizontalGlue(), cbLog, Box.createHorizontalGlue(), btTexture, Box.createHorizontalGlue(), btSVM, Box.createHorizontalGlue(), btStop, Box.createHorizontalGlue());
		mainPanel.add(p4);
	}

	@Override
	public void stopInterface() {
		XMLPreferences preferences = getPreferences();

		preferences.putInt("w", Integer.parseInt(prmW.getText()));
		preferences.putInt("p", Integer.parseInt(prmP.getText()));
		preferences.putDouble("r", Double.parseDouble(prmR.getText()));
		preferences.putDouble("t", Double.parseDouble(prmT.getText()));
		preferences.putDouble("c", Double.parseDouble(prmC.getText()));
		preferences.putDouble("g", Double.parseDouble(prmG.getText()));
		// preferences.putInt("v", Integer.parseInt(prmV.getText()));
	}

	@Override
	public Dimension getDefaultFrameDimension() {
		return null;
	}
}
