"""Build the IEEE paper: paper/fbimpact_ieee.tex (IEEEtran, for submission) and paper/fbimpact_ieee.html (for the PDF).

    python paper/build_paper.py

Every number in the text comes from the verified run records (scripts/verify_results.py, results/analysis/*.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperlib import Doc  # noqa: E402

HERE = Path(__file__).resolve().parent

META = {
    "title": "Anticipating Falls from Skeletal Pose: A Pre-Registered, Held-Out-Domain Evaluation of Causal ST-GCN Warnings and Their Joint Evidence",
    "title_plain": "Anticipating Falls from Skeletal Pose: A Pre-Registered, Held-Out-Domain Evaluation of Causal ST-GCN Warnings and Their Joint Evidence",
    "running_head": "Sahoo: Anticipating Falls from Skeletal Pose",
    "authors_tex": (
        r"\IEEEauthorblockN{Aditya~Ayushman~Sahoo}"
        r"\IEEEauthorblockA{Department of Computational Intelligence\\School of Computing\\"
        r"SRM Institute of Science and Technology\\Kattankulathur -- 603203\\adityaasahoo@gmail.com}"
        r"\and"
        r"\IEEEauthorblockN{Kothai~G}"
        r"\IEEEauthorblockA{Department of Computational Intelligence\\School of Computing\\"
        r"Faculty of Engineering and Technology\\SRM Institute of Science and Technology\\Kattankulathur -- 603203}"
    ),
    "authors_html": (
        '<div class="au-block"><div class="an">Aditya Ayushman Sahoo</div><div class="aa">'
        'Department of Computational Intelligence<br>School of Computing<br><br>'
        'SRM Institute of Science and Technology<br>Kattankulathur &ndash; 603203<br>adityaasahoo@gmail.com</div></div>'
        '<div class="au-block"><div class="an">Kothai G</div><div class="aa">'
        'Department of Computational Intelligence<br>School of Computing<br>Faculty of Engineering and Technology<br>'
        'SRM Institute of Science and Technology<br>Kattankulathur &ndash; 603203</div></div>'
    ),
    "abstract": (
        "Pre-impact fall detection aims to raise a warning before the body reaches the ground, leaving time for a protective response such as an airbag or a call for help. "
        "Most evidence comes from wearable inertial sensors, and camera-based work is scarce and usually evaluated on windows cut at a fixed lead time rather than on streaming video. "
        "This paper studies fall anticipation from 2-D skeletal pose with a causal spatio-temporal graph convolutional network (ST-GCN) that outputs a per-frame imminence score, "
        "a threshold-and-persistence rule that issues warnings, and a deletion/insertion test of the joints named at each warning. "
        "Decision rules for all confirmatory comparisons were committed before the corresponding runs. "
        "On two public datasets of acted falls (UR Fall and Le2i; 200 clips, 129 falls), five-fold clip-independent evaluation gave a mean lead time of 0.41 to 0.49 s and pooled recall of 0.63 to 0.68, "
        "at 176 to 246 false alarms per hour at the default operating point. "
        "Neither design hypothesis was supported: the time-weighted pre-impact loss lowered UR Fall recall by 0.133 (95% CI 0.067 to 0.211), "
        "and four-phase supervision did not meet the pre-registered false-alarm criterion over four fresh seeds. "
        "Recall ranged from 0.03 to 0.97 when a room or dataset was held out, against 0.64 in-domain. "
        "Joint relevance exceeded a random joint order (deletion gap 0.043, insertion gap 0.072; 90 runs). "
        "We discuss what these results imply for evaluating camera-based pre-impact detection."
    ),
    "keywords": "Fall detection, pre-impact detection, skeleton-based action recognition, graph convolutional networks, explainable AI, dataset shift, pre-registration",
    # Removed for now at the author's request; re-add before submission — IEEE requires a generative-AI
    # disclosure if one applies. Previous text:
    # "Generative-AI disclosure, in line with IEEE policy: an AI assistant (Claude, Anthropic) was used to help draft the text of this article, "
    # "to write and review the analysis, figure and verification scripts in the accompanying repository, and to check reported numbers against the run records. "
    # "The experiments were designed and run by the author, who reviewed all content and is responsible for it."
    "acknowledgment": "",
}

REFS = {
    "tamura": 'T. Tamura, T. Yoshimura, M. Sekine, M. Uchida, and O. Tanaka, “A wearable airbag to prevent fall injuries,” <i>IEEE Trans. Inf. Technol. Biomed.</i>, vol. 13, no. 6, pp. 910–914, Nov. 2009, doi: 10.1109/TITB.2009.2033673.',
    "kfall": 'X. Yu, J. Jang, and S. Xiong, “A large-scale open motion dataset (KFall) and benchmark algorithms for detecting pre-impact fall of the elderly using wearable inertial sensors,” <i>Front. Aging Neurosci.</i>, vol. 13, Art. no. 692865, Jul. 2021, doi: 10.3389/fnagi.2021.692865.',
    "prefallkd": 'T.-H. Chi, K.-C. Liu, C.-Y. Hsieh, Y. Tsao, and C.-T. Chan, “PreFallKD: Pre-impact fall detection via CNN-ViT knowledge distillation,” 2023, arXiv:2303.03634.',
    "jesudhas": 'P. Jesudhas, T. Raghuveera, and S. Jeyaraj, “Computationally efficient non-intrusive pre-impact fall detection system,” 2025, arXiv:2507.03705.',
    "shin": 'J. Shin <i>et al.</i>, “Computer-aided fall recognition using a three-stream spatial-temporal GCN model with adaptive feature aggregation,” 2024, arXiv:2408.12211.',
    "urfd": 'B. Kwolek and M. Kepski, “Human fall detection on embedded platform using depth maps and wireless accelerometer,” <i>Comput. Methods Programs Biomed.</i>, vol. 117, no. 3, pp. 489–501, Dec. 2014, doi: 10.1016/j.cmpb.2014.09.005.',
    "le2i": 'I. Charfi, J. Miteran, J. Dubois, M. Atri, and R. Tourki, “Optimised spatio-temporal descriptors for real-time fall detection: Comparison of support vector machine and Adaboost-based classification,” <i>J. Electron. Imaging</i>, vol. 22, no. 4, Art. no. 041106, 2013, doi: 10.1117/1.JEI.22.4.041106.',
    "upfall": 'L. Martínez-Villaseñor, H. Ponce, J. Brieva, E. Moya-Albor, J. Núñez-Martínez, and C. Peñafort-Asturiano, “UP-Fall detection dataset: A multimodal approach,” <i>Sensors</i>, vol. 19, no. 9, Art. no. 1988, Apr. 2019, doi: 10.3390/s19091988.',
    "stgcn": 'S. Yan, Y. Xiong, and D. Lin, “Spatial temporal graph convolutional networks for skeleton-based action recognition,” in <i>Proc. AAAI Conf. Artif. Intell.</i>, 2018, arXiv:1801.07455.',
    "adam": 'D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2015, arXiv:1412.6980.',
    "sgdr": 'I. Loshchilov and F. Hutter, “SGDR: Stochastic gradient descent with warm restarts,” in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2017, arXiv:1608.03983.',
    "rise": 'V. Petsiuk, A. Das, and K. Saenko, “RISE: Randomized input sampling for explanation of black-box models,” in <i>Proc. Brit. Mach. Vis. Conf. (BMVC)</i>, 2018, arXiv:1806.07421.',
    "samek": 'W. Samek, A. Binder, G. Montavon, S. Lapuschkin, and K.-R. Müller, “Evaluating the visualization of what a deep neural network has learned,” <i>IEEE Trans. Neural Netw. Learn. Syst.</i>, vol. 28, no. 11, pp. 2660–2673, Nov. 2017.',
    "jain": 'S. Jain and B. C. Wallace, “Attention is not explanation,” in <i>Proc. NAACL-HLT</i>, 2019, pp. 3543–3556, doi: 10.18653/v1/N19-1357.',
    "wiegreffe": 'S. Wiegreffe and Y. Pinter, “Attention is not not explanation,” in <i>Proc. EMNLP-IJCNLP</i>, 2019, pp. 11–20, doi: 10.18653/v1/D19-1002.',
    "adebayo": 'J. Adebayo, J. Gilmer, M. Muelly, I. Goodfellow, M. Hardt, and B. Kim, “Sanity checks for saliency maps,” in <i>Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 31, 2018.',
    "nosek": 'B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, “The preregistration revolution,” <i>Proc. Natl. Acad. Sci. USA</i>, vol. 115, no. 11, pp. 2600–2606, Mar. 2018, doi: 10.1073/pnas.1708274114.',
    "efron": 'B. Efron, “Bootstrap methods: Another look at the jackknife,” <i>Ann. Statist.</i>, vol. 7, no. 1, pp. 1–26, 1979.',
    "torralba": 'A. Torralba and A. A. Efros, “Unbiased look at dataset bias,” in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2011, pp. 1521–1528.',
}

BLOCKS: list = []
add = BLOCKS.append

# ============================================================================================ I. Introduction
add(("sec", "Introduction", "sec:intro"))
add(("p", "A fall that is recognised only after the body reaches the floor can summon help, but it cannot protect the body during the fall. "
          "Devices such as wearable airbags have to act before contact, and an early prototype inflated its airbag on a trigger issued about 300 ms before impact[@tamura]. "
          "Pre-impact detection has therefore been studied mostly with wearable inertial sensors. "
          "The KFall benchmark reports mean lead times of 333 to 403 ms for three detectors[@kfall], and a distilled network evaluated on the same data reports 551 ms[@prefallkd]."))
add(("p", "A camera needs nothing to be worn or charged, but video is often unacceptable in private rooms. "
          "A pose skeleton discards the pixels and keeps the motion, which makes it an attractive intermediate representation. "
          "Graph-convolutional skeleton models now report accuracies above 99% for fall versus non-fall classification on UR Fall and Le2i[@shin], but that is a recognition task. "
          "Anticipation from cameras has received far less attention. The most closely related study we are aware of classifies short skeleton windows cut at a fixed lead time before impact[@jesudhas]."))
add(("p", "Three features of current evaluation practice limit what such results say about deployment. "
          "First, windows cut at a fixed lead make lead time a property of the input rather than an outcome, whereas a deployed system streams, "
          "issues warnings at times of its own choosing, and is charged for every warning during normal activity. "
          "Second, accuracy and F1 on balanced sets hide the alarm rate per hour of ordinary life, which decides whether a system can be tolerated. "
          "Third, random or unspecified splits put clips from the same rooms and cameras in both training and test sets, so they cannot reveal sensitivity to dataset shift[@torralba]."))
add(("p", "This paper evaluates a causal ST-GCN on two public datasets under a streaming, pre-registered protocol. We ask four questions."))
add(("enum", [
    "<b>RQ1.</b> Can a lightweight skeleton model warn before impact with useful lead time at high recall, and at what false-alarm rate?",
    "<b>RQ2.</b> Are the joints named at each warning faithful to what drives the model’s score?",
    "<b>RQ3.</b> Does an explicit pre-impact objective, or auxiliary phase supervision, improve anticipation?",
    "<b>RQ4.</b> Do in-domain results survive the removal of a room or a dataset from training?",
]))
add(("p", "The contributions are as follows."))
add(("enum", [
    "A streaming evaluation protocol for fall anticipation in which lead time is measured on warned falls, a warning before the labelled imminent window counts as a false alarm, and operating points are frozen on validation clips.",
    "Pre-registered confirmatory tests, on four fresh seeds, of two design hypotheses that looked promising on a single seed. Neither is supported, and the time-weighted loss lowers recall on UR Fall.",
    "A held-out-domain evaluation (24 runs) in which recall ranges from 0.03 to 0.97 when a room or dataset is withheld.",
    "A deletion/insertion evaluation of the joints named at each warning over 90 runs, analysed with a run-level bootstrap.",
    "A scripted audit that recomputes every reported number from the saved run folders.",
]))
add(("p", "Section [#sec:related] reviews related work, Section [#sec:methods] describes the data, model and protocol, Section [#sec:results] reports the results, "
          "and Sections [#sec:discussion] and [#sec:conclusion] discuss their implications and limits."))

# ============================================================================================ II. Related work
add(("sec", "Related Work", "sec:related"))
add(("sub", "Post-Fall Detection on UR Fall and Le2i", "sub:post"))
add(("p", "UR Fall contains 70 sequences, 30 of falls and 40 of activities of daily living, recorded with Kinect cameras[@urfd]. "
          "Le2i was recorded in home-like environments[@le2i]. Recent skeleton-based graph models report very high accuracy on both. "
          "Shin <i>et al.</i> report 99.15% on UR Fall and 99.64% on ImViA/Le2i for a three-stream ST-GCN with adaptive feature aggregation[@shin]. "
          "In the same paper the fall class on UR Fall has a sensitivity of 86.67%, and frames without an extractable skeleton (853 of 2,995 on UR Fall) were discarded. "
          "The text does not state the splitting protocol, and the task is classification of fall against non-fall rather than anticipation."))
add(("sub", "Pre-Impact Detection", "sub:pre"))
add(("p", "<b>Wearable sensors.</b> Yu <i>et al.</i> released KFall (32 participants, 21 daily activities and 15 simulated falls, low-back inertial sensor, fall times labelled from synchronised video) "
          "and benchmarked three detectors on a subject-wise split (26 training and 6 test subjects; 444 fall and 507 non-fall test files)[@kfall]. "
          "Lead time is the interval between the detection moment and impact. A ConvLSTM reached sensitivity 99.32%, specificity 99.01% and a mean lead time of 403 ± 163 ms, "
          "against 385 ± 159 ms for an SVM and 333 ± 160 ms for a threshold rule. A knowledge-distillation model on the same dataset reports an F1-score of 92.66% and a lead time of 551.3 ms[@prefallkd]."))
add(("p", "<b>Cameras.</b> Jesudhas <i>et al.</i> apply an LSTM to a small set of fall-specific features of 2-D skeleton joints (MoveNet) on UP-Fall[@upfall] and describe the work as the first attempt at pre-impact detection from CCTV imagery[@jesudhas]. "
          "Windows of 15 frames are drawn at a fixed lead before impact and classified. With a random 80:20 split they report a macro-F1 of 0.88 at a 0.5 s lead "
          "(pre-impact class: precision 0.93, recall 0.72), falling from 0.93 at 0.1 s to 0.69 at 0.8 s. [#tab:related] summarises these studies next to the present one."))
add(("table", dict(
    label="tab:related", wide=True, spec=r"p{0.13\textwidth}p{0.13\textwidth}p{0.14\textwidth}p{0.15\textwidth}p{0.29\textwidth}p{0.11\textwidth}",
    caption="Closest published work. The studies differ in modality, data, task formulation, split and metric, so their figures describe each setting and are not ranked against one another.",
    header=["Study", "Input", "Data", "Task", "Reported result", "Split"],
    rows=[
        ["Tamura <i>et al.</i>[@tamura]", "Wearable accelerometer and gyroscope", "16 subjects, simulated falls", "Threshold trigger for an airbag", "Trigger about 300 ms before impact", "Not applicable"],
        ["Yu <i>et al.</i>[@kfall]", "Wearable IMU (low back)", "KFall, 32 participants", "Pre-impact detection", "ConvLSTM: sensitivity 99.32%, specificity 99.01%, lead 403 ± 163 ms", "Subject-wise, 26/6"],
        ["Chi <i>et al.</i>[@prefallkd]", "Wearable IMU", "KFall", "Pre-impact detection", "F1 92.66%, lead 551.3 ms", "Not examined (preprint)"],
        ["Jesudhas <i>et al.</i>[@jesudhas]", "2-D skeleton (MoveNet), camera", "UP-Fall, 17 subjects", "Classify 15-frame windows at a fixed lead", "Macro-F1 0.88 at 0.5 s; fall-class recall 0.72", "Random 80:20"],
        ["Shin <i>et al.</i>[@shin]", "2-D skeleton, three-stream ST-GCN", "UR Fall, ImViA, UP-Fall, FU-Kinect", "Fall versus non-fall", "Accuracy 99.15% (UR Fall), 99.64% (ImViA)", "Not stated in text"],
        ["<b>This work</b>", "2-D skeleton, causal ST-GCN", "UR Fall + Le2i, 200 clips", "Streaming anticipation, frozen threshold", "Recall 0.63–0.68, lead 0.41–0.44 s, 176–246 false alarms per hour", "Clip-independent 5-fold; four held-out domains"],
    ])))
add(("sub", "Skeleton Networks", "sub:stgcn"))
add(("p", "ST-GCN applies graph convolution over the body graph and temporal convolution over frames[@stgcn]. "
          "The present model is a causal variant: temporal convolutions pad on the left only and every derived feature uses backward differences, "
          "so the network itself cannot look ahead. The offline pose normalisation that precedes it is the one non-causal step, discussed in Section [#sec:discussion]. "
          "We train it with Adam[@adam] and a cosine learning-rate schedule[@sgdr]."))
add(("sub", "Explanation Faithfulness", "sub:faith"))
add(("p", "Deletion and insertion curves were introduced for image classifiers in RISE[@rise]; related region-perturbation measures were studied earlier[@samek]. "
          "Both ask whether removing the inputs an explanation ranks highest hurts the score more than removing random ones. "
          "Whether attention weights are explanations at all is contested: Jain and Wallace show that they often disagree with gradient-based importance[@jain], "
          "Wiegreffe and Pinter argue that the answer depends on the definition and the test used[@wiegreffe], and saliency maps in general can fail basic sanity checks[@adebayo]. "
          "We treat the question as empirical for one architecture."))
add(("sub", "Reporting Practice", "sub:report"))
add(("p", "Mistaking the generation of a hypothesis for its test lowers the credibility of a finding, and hindsight bias makes the mistake hard to avoid[@nosek]. "
          "Model-comparison studies in this area rarely guard against it. Here, the decision rules for the confirmation grid and for the held-out-domain protocols were committed before the first run each governs "
          "(Section [#sec:methods]), and intervals come from a paired bootstrap[@efron]."))

# ============================================================================================ III. Methods
add(("sec", "Materials and Methods", "sec:methods"))
add(("p", "[#tab:notation] lists the symbols used in this section."))
add(("table", dict(
    label="tab:notation", wide=False, spec=r"p{0.27\columnwidth}p{0.65\columnwidth}",
    caption="Notation.",
    header=["Symbol", "Meaning"],
    rows=[
        ["<m>t^{*}</m>", "Impact frame (first ground contact)"],
        ["<m>W</m>, <m>D</m>", "Imminent window in frames; in seconds, <m>D=W/f_s</m>"],
        ["<m>f_s</m>", "Frame rate of a clip"],
        ["<m>C, T, V</m>", "Channels, window length and joints (14, 25, 17)"],
        ["<m>p_t</m>", "Per-frame imminence probability"],
        ["<m>\\tau</m>, <m>k</m>, <m>R</m>", "Score threshold, persistence and refractory period (frames)"],
        ["<m>t_{\\mathrm{warn}}</m>, <m>L</m>", "Warning frame and lead time"],
        ["<m>\\lambda</m>", "Pre-impact time-weighting strength"],
        ["<m>A_k</m>, <m>M_k</m>", "Adjacency of partition <m>k</m> and its learnable edge importance"],
        ["<m>a_{t,v}</m>", "Joint attention weight at frame <m>t</m>, joint <m>v</m>"],
        ["<m>\\Delta_{\\mathrm{del}}</m>, <m>\\Delta_{\\mathrm{ins}}</m>", "Deletion and insertion faithfulness gaps"],
    ])))

add(("sub", "System Overview", "sub:overview"))
add(("p", "[#fig:pipeline] shows the processing chain. A frozen pose estimator converts video into 17-keypoint skeletons, after which pixels are discarded, so only skeletons are stored and modelled. "
          "Skeletons are normalised, enriched with motion features and cut into windows. A spatio-temporal graph network scores every frame with an imminence probability, "
          "a threshold-and-persistence rule turns the score stream into warnings with a lead time, and at each warning the relevance of every joint is extracted and tested by deletion and insertion."))
add(("fig", dict(label="fig:pipeline", file="fig1_pipeline", wide=True,
                 caption="Processing chain. A frozen pose estimator turns video into skeletons, after which pixels are discarded. A causal ST-GCN scores every frame, a threshold and persistence rule issues the warning, and the joint relevance at the warning frame is scored by deletion and insertion.")))

add(("sub", "Datasets", "sub:data"))
add(("p", "We use UR Fall (70 sequences, camera 0) and the Home and Coffee-room scenes of Le2i (130 clips), pooled into a combined benchmark of 200 clips, 129 falls and 71 normal clips ([#tab:data]). "
          "Every clip carries its own frame rate and all frame counts are converted to seconds, so pooling clips recorded at 24 to 30 frames per second is safe. "
          "The impact frame <i>t</i><sup>*</sup> comes from each dataset’s own annotation (the first frame labelled “lying” on UR Fall, the annotated fall end on Le2i) and never from the skeleton the model reads, so lead time is not circular. "
          "The UR Fall label falls at or after ground contact, which shortens measured lead time. "
          "A procedurally generated fixture of 320 clips was used only to test the code and is excluded from every result. "
          "Neither dataset publishes actor identity, so splits are independent by clip, not by subject, and may overstate performance on unseen people."))
add(("table", dict(
    label="tab:data", wide=True, spec=r"lccc",
    caption="Datasets after skeleton extraction.",
    header=["Property", "UR Fall", "Le2i (Home, Coffee room)", "Combined"],
    rows=[
        ["Clips (falls / normal)", "70 (30 / 40)", "130 (99 / 31)", "200 (129 / 71)"],
        ["Duration; normal activity", "6.6 min; 5.8 min", "28.6 min; 20.6 min", "35.2 min; 26.4 min"],
        ["Frame rate", "30 fps", "24 to 25 fps", "24 to 30 fps"],
        ["Frames with a usable skeleton", "79.8%", "85.3%", "n/a"],
        ["Impact frame", "first frame labelled “lying”", "annotated fall end", "dataset annotation"],
        ["Frames before impact (min / median / max)", "33 / 56 / 177", "1 / 161 / 1,115", "n/a"],
        ["Frames per phase (normal / imminent / falling / grounded)", "9,664 / 469 / 930 / 873", "28,178 / 1,632 / 2,289 / 9,967", "37,842 / 2,101 / 3,219 / 10,840"],
    ])))

add(("sub", "Pose and Features", "sub:pose"))
add(("p", "Skeletons follow the COCO 17-keypoint layout from a frozen, off-the-shelf pose estimator (Ultralytics YOLO11n-pose). "
          "Joints below a confidence of 0.30 (0.35 for the UR Fall ablations) are treated as missing and linearly interpolated. "
          "Coordinates are centred on the mid-hip and divided by the clip’s median torso length, so they are independent of where the person stands and how far away the camera is. "
          "[#tab:features] defines the channels. The UR Fall ablations use position and velocity <m>(x, y, v_x, v_y)</m>; the combined-benchmark arms use either the same four channels (arm A) or all 14 (arms B–D). "
          "Velocities and accelerations are backward differences, clipped to loose bounds so that a single pose-estimator dropout cannot set a channel’s scale. "
          "In the offline evaluation, however, interpolation uses the observed frames on both sides of a gap and the scale is the median torso length over the clip, "
          "whereas a live system can only hold the last value and track a running median."))
add(("table", dict(
    label="tab:features", wide=True, spec=r"p{0.15\textwidth}p{0.06\textwidth}p{0.69\textwidth}",
    caption="The 14 input channels. Arm A and the UR Fall ablations use the first four; arms B–D use all 14. Derivatives are per second in torso lengths.",
    header=["Channels", "Count", "Definition"],
    rows=[
        ["<m>x, y</m>", "2", "Joint position after centring on the mid-hip and scaling by the median torso length"],
        ["<m>v_x, v_y</m>", "2", "Backward-difference velocity, clipped to ±30 torso lengths per second"],
        ["<m>a_x, a_y</m>", "2", "Backward-difference acceleration, clipped to ±300"],
        ["<m>b_x, b_y</m>", "2", "Bone vector: offset from the joint to its parent in a fixed spanning tree (the mid-hip and mid-shoulder act as parents, which keeps the features mirror-symmetric)"],
        ["<m>\\dot b_x, \\dot b_y</m>", "2", "Backward-difference rate of the bone vector"],
        ["<m>\\theta, \\dot\\theta</m>", "2", "Trunk angle from vertical (sign dropped, so forward and backward falls look alike) and its rate; shared by all joints"],
        ["<m>h, \\dot h</m>", "2", "Mid-hip height as a whole-body summary, and its vertical rate; shared by all joints"],
    ])))

add(("sub", "Network Architecture", "sub:model"))
add(("p", "The network ([#fig:arch]) maps a window <m>X \\in \\mathbb{R}^{C\\times T\\times V}</m> to one logit per frame. The input channels are batch-normalised and passed through four ST-GCN blocks with 64, 64, 128 and 128 channels. "
          "Each block first applies a spatial graph convolution over the one-hop neighbourhood of every joint, split into <m>K=3</m> partitions: the joint itself, neighbours nearer the body centre (the hips) and neighbours farther from it[@stgcn],"))
add(("eq", "eq:gconv", r"H^{(l+1)} = \sum_{k=1}^{3} \left(A_k \odot M_k\right) H^{(l)} W_k,",
     "<span class=\"m\">H<sup>(l+1)</sup> = &Sigma;<sub>k=1..3</sub> (A<sub>k</sub> &odot; M<sub>k</sub>) H<sup>(l)</sup> W<sub>k</sub></span>"))
add(("p", "where <m>A_k</m> are row-normalised adjacency matrices, <m>M_k</m> is a learnable edge-importance matrix initialised to ones, and <m>W_k</m> is a <m>1\\times 1</m> convolution. "
          "A temporal convolution of kernel size <m>\\kappa = 7</m> then pads only the past, so that the output at frame <m>t</m> reads frames <m>t-6,\\dots,t</m> only,"))
add(("eq", "eq:tconv", r"h'_t = \sum_{j=0}^{\kappa-1} U_j\, h_{t-j},",
     "<span class=\"m\">h&prime;<sub>t</sub> = &Sigma;<sub>j=0..&kappa;&minus;1</sub> U<sub>j</sub> h<sub>t&minus;j</sub></span>"))
add(("p", "followed by batch normalisation, dropout (0.1), a residual connection and a ReLU. With four blocks of kernel 7 and dilation 1 the receptive field is <m>1+\\sum_l(\\kappa-1)d_l = 25</m> frames, equal to the input window on the combined benchmark (30 frames on UR Fall). "
          "A joint-attention module then weights the joints of each frame from that frame’s own features, which keeps the weights causal and attributable to a single frame–joint pair,"))
add(("eq", "eq:attn", r"a_{t,v}=\frac{\exp\!\big(w^{\top}\tanh(U h_{t,v})\big)}{\sum_{u}\exp\!\big(w^{\top}\tanh(U h_{t,u})\big)},\qquad z_t=\sum_{v}a_{t,v}\,h_{t,v}.",
     "<span class=\"m\">a<sub>t,v</sub> = exp(w<sup>T</sup>tanh(U h<sub>t,v</sub>)) / &Sigma;<sub>u</sub> exp(w<sup>T</sup>tanh(U h<sub>t,u</sub>)), &nbsp;z<sub>t</sub> = &Sigma;<sub>v</sub> a<sub>t,v</sub> h<sub>t,v</sub></span>"))
add(("p", "A <m>1\\times 1</m> convolution maps <m>z_t</m> to the imminence logit and <m>p_t</m> is its sigmoid. The attention weights are what we report as the evidence for each warning (we call the module the grounding module). "
          "The model has about 0.4 million parameters. An optional four-way phase head (normal, imminent, falling, grounded) sits beside the imminence head and adds auxiliary supervision. "
          "The ablation variants change one component each: “without grounding” replaces the attention pooling by a mean over joints, “without temporal modelling” sets the temporal kernel to 1 and averages the window into a single score, "
          "and “without velocity” removes the velocity channels."))
add(("fig", dict(label="fig:arch", file="fig2_architecture", wide=True,
                 caption="Network architecture. (a) Block diagram: each ST-GCN block combines a spatial graph convolution over three partitions with a causal temporal convolution; joint attention pools the joints and a 1 × 1 head emits one logit per frame. (b) The 17-joint body graph with the hips as its centre. (c) Receptive field of the four causal blocks: the output at frame t reads only the previous 25 frames.")))

add(("sub", "Labels and Objective", "sub:loss"))
add(("p", "With <i>W</i> the length of the imminent window in frames (17 frames, 0.68 s, on the combined benchmark; 20 frames, 0.67 s, on UR Fall), the binary target is"))
add(("eq", "eq:label", r"y_t = \begin{cases} 1, & t^{*}-W \le t \le t^{*},\\ 0, & t < t^{*}-W \ \text{or the clip is normal},\\ \text{ignored}, & t > t^{*}, \end{cases}",
     "<span class=\"m\">y<sub>t</sub> = 1 if t*&minus;W &le; t &le; t*; &nbsp;0 if t &lt; t*&minus;W or the clip is normal; &nbsp;ignored if t &gt; t*</span>"))
add(("p", "The per-frame loss is class-weighted binary cross-entropy with positive weight <m>\\rho = \\min\\{(1-\\pi)/\\pi,\\,20\\}</m>, where <m>\\pi</m> is the positive frame fraction in the training split. "
          "The pre-impact objective scales each frame inside the imminent window by an exponential in the time to impact <m>d_t = (t^{*}-t)/f_s</m>,"))
add(("eq", "eq:weight", r"w_t = \frac{e^{\lambda d_t}}{Z},\qquad Z=\frac{e^{\lambda D}-1}{\lambda D},\qquad 0\le d_t\le D=\frac{W}{f_s},",
     "<span class=\"m\">w<sub>t</sub> = e<sup>&lambda;d<sub>t</sub></sup> / Z, &nbsp;Z = (e<sup>&lambda;D</sup>&minus;1) / (&lambda;D), &nbsp;0 &le; d<sub>t</sub> &le; D = W / f<sub>s</sub></span>"))
add(("p", "and <m>w_t = 1</m> elsewhere. <m>Z</m> normalises the mean weight over the window to one, so <m>\\lambda</m> can change without changing the effective learning rate, and <m>\\lambda = 0</m> recovers the plain loss. "
          "With <m>m_t</m> the mask of non-ignored frames and <m>\\ell_t</m> the weighted cross-entropy of frame <m>t</m>,"))
add(("eq", "eq:loss", r"\mathcal{L} = \frac{\sum_t m_t w_t \ell_t}{\sum_t m_t w_t} + 0.5\,\mathcal{L}_{\mathrm{phase}} + 0.1\,\mathcal{L}_{\mathrm{cons}},",
     "<span class=\"m\"><span class=\"cal\">L</span> = &Sigma;<sub>t</sub> m<sub>t</sub>w<sub>t</sub>&ell;<sub>t</sub> / &Sigma;<sub>t</sub> m<sub>t</sub>w<sub>t</sub> + 0.5 <span class=\"cal\">L</span><sub>phase</sub> + 0.1 <span class=\"cal\">L</span><sub>cons</sub></span>"))
add(("p", "where the last two terms exist only for the phase arms. The four phases are normal, imminent (the <i>W</i> frames before the annotated fall onset), falling (onset to impact) and grounded (after impact); "
          "<m>\\mathcal{L}_{\\mathrm{phase}}</m> is a cross-entropy over them with capped inverse-frequency class weights, "
          "and <m>\\mathcal{L}_{\\mathrm{cons}}</m> is the mean squared difference between <m>p_t</m> and the phase head’s probability mass on the imminent and falling phases. "
          "The full UR Fall model uses <m>\\lambda = 1.5</m>; all combined-benchmark arms use <m>\\lambda = 0</m> because the UR Fall result below showed no benefit from the weighting."))

add(("sub", "Decision Rule and Operating-Point Selection", "sub:decision"))
add(("p", "[#alg:stream] turns the score stream into warnings. A warning fires at the <i>k</i>-th consecutive frame at which the score reaches a threshold <m>\\tau</m>, "
          "and a refractory period of <m>R</m> frames (25 on the combined benchmark, 30 on UR Fall) suppresses repeat alarms from one episode. "
          "The warning time <m>t_{\\mathrm{warn}}</m> is the <i>k</i>-th frame, because the system cannot know a run of <i>k</i> frames occurred earlier. Lead time is"))
add(("algo", dict(
    label="alg:stream", caption="Streaming warning generation (one clip).",
    lines=[
        ("req", "scores <m>p_1, p_2, \\dots</m> arriving one frame at a time; threshold <m>\\tau</m>; persistence <m>k</m>; refractory period <m>R</m>"),
        ("state", "<m>\\text{run} \\leftarrow 0</m>; <m>\\text{cooldown} \\leftarrow 0</m>"),
        ("for", "each new frame <m>t</m>"),
        ("if", "<m>p_t \\ge \\tau</m>"),
        ("state", "<m>\\text{run} \\leftarrow \\text{run}+1</m>"),
        ("else", ""),
        ("state", "<m>\\text{run} \\leftarrow 0</m>"),
        ("endif", ""),
        ("if", "<m>\\text{cooldown} > 0</m>"),
        ("state", "<m>\\text{cooldown} \\leftarrow \\text{cooldown}-1</m> // suppressed"),
        ("elsif", "<m>\\text{run} \\ge k</m>"),
        ("state", "emit a warning with <m>t_{\\mathrm{warn}} \\leftarrow t</m>"),
        ("state", "<m>\\text{cooldown} \\leftarrow R</m>"),
        ("endif", ""),
        ("endfor", ""),
    ])))
add(("eq", "eq:lead", r"L = (t^{*} - t_{\mathrm{warn}})/f_s,",
     "<span class=\"m\">L = (t* &minus; t<sub>warn</sub>) / f<sub>s</sub></span>"))
add(("p", "reported over correctly warned falls only. A warning inside the window <m>[t^{*}-W,\\,t^{*}]</m> counts as a hit, and the earliest one sets the lead time. "
          "A fall with no such warning is a miss. Any warning in a normal clip, or before the imminent window in a fall clip, is a false alarm, because that frame is labelled normal; warnings after impact are neither credited nor penalised. "
          "Recall is pooled (warned falls divided by tested falls, summed over folds), and false alarms are counted and also expressed per hour of normal activity."))
add(("p", "The pair <m>(\\tau, k)</m> is chosen on validation clips only ([#alg:select]) and frozen into the checkpoint, so the test set never influences it. "
          "The same validation score selects the training epoch: the mean lead time at the selected operating point, or the frame-level AUC minus one when no point meets the budget."))
add(("algo", dict(
    label="alg:select", caption="Operating-point selection on validation clips.",
    lines=[
        ("req", "validation clips with scores; grid <m>\\tau \\in \\{0.05, 0.10, \\dots, 0.95\\}</m>, <m>k \\in \\{1,2,3,5,8\\}</m>; false-alarm budget <m>B</m> (2 per hour)"),
        ("ens", "operating point <m>(\\tau^{*}, k^{*})</m>, or a default if none is feasible"),
        ("state", "<m>S \\leftarrow \\emptyset</m>"),
        ("for", "each <m>(\\tau, k)</m> in the grid"),
        ("state", "run [#alg:stream] on every validation clip; compute recall, mean lead time <m>\\bar L</m> and false alarms per hour"),
        ("if", "false alarms per hour <m>\\le B</m> and at least one fall is warned"),
        ("state", "add <m>(\\tau, k)</m> to <m>S</m>"),
        ("endif", ""),
        ("endfor", ""),
        ("if", "<m>S = \\emptyset</m>"),
        ("return", "default <m>(0.70, 3)</m> and validation score AUC <m>-1</m>"),
        ("endif", ""),
        ("return", "the point of <m>S</m> that maximises <m>\\bar L</m>, ties broken by recall"),
    ])))

add(("sub", "Evaluation Protocols", "sub:protocol"))
add(("p", "[#fig:protocol] summarises the workflow. In-domain evaluation uses five-fold cross-validation over clips, with folds fixed by a split seed so that every arm and every training seed scores the same clips and contrasts are paired. "
          "Held-out-domain evaluation uses four protocols that remove a place or dataset from training: UR Fall to Le2i, Le2i to UR Fall, leave the Coffee room out, and leave the Home scene out. "
          "Validation clips come from the training side only, and the test domain influences no choice. [#tab:settings] gives the training settings of the confirmation grid."))
add(("fig", dict(label="fig:protocol", file="fig3_protocol", wide=True,
                 caption="Evaluation workflow. The in-domain grid and the held-out-domain protocols share training, operating-point selection and the streaming test; the decision rules and analysis plan were committed before the first run they govern.")))
add(("table", dict(
    label="tab:settings", wide=True, spec=r"p{0.17\textwidth}p{0.30\textwidth}p{0.18\textwidth}p{0.27\textwidth}",
    caption="Training and protocol settings of the combined-benchmark confirmation grid.",
    header=["Setting", "Value", "Setting", "Value"],
    rows=[
        ["Window; stride", "25 frames; 4 (training), 1 (inference)", "Imminent window <i>W</i>", "17 frames (0.68 s)"],
        ["Blocks; kernel; dropout", "64-64-128-128; 7; 0.1", "Pre-impact weight <m>\\lambda</m>", "0 (arms A–D); 1.5 (UR Fall full model)"],
        ["Optimiser", "Adam, learning rate 0.001, weight decay 0.0001", "Phase and consistency weights", "0.5; 0.1 (arms C, D); 0 (D consistency)"],
        ["Schedule", "Cosine, 2 warm-up epochs, gradient clip 5", "False-alarm budget; refractory", "2 per hour; 25 frames"],
        ["Batch; epochs; patience", "24; 30; 6 (evaluated every 2 epochs)", "Explanations", "At most 40 warnings per run; 5 random orders; zero baseline"],
        ["Precision", "Mixed (automatic)", "Augmentation", "Noise <m>\\sigma</m> = 0.015; scale 0.85–1.15; flip 0.5"],
    ])))

add(("sub", "Faithfulness Test", "sub:ftest"))
add(("p", "At each warning the joints are ranked by relevance (the attention weights of the warning frame; gradient × input for one ablation). [#alg:faith] removes joints in that order: "
          "a removed joint has all its channels set to zero in every frame of the window, and the score is read at the warning frame, giving <m>V+1=18</m> points per curve. "
          "Insertion starts from a blank skeleton and adds the joints in the same order. Each curve is compared with the mean of five random orders. "
          "Removing joints produces skeletons the model never saw in training, so part of any score drop is distribution shift; the random-order control suffers the same shift and is subtracted. "
          "With <m>A(\\cdot)</m> the trapezoidal area under a curve, normalised by the number of steps, the gaps are"))
add(("algo", dict(
    label="alg:faith", caption="Deletion and insertion faithfulness for one warning.",
    lines=[
        ("req", "model <m>f</m>; window <m>X</m>; warning frame <m>t</m>; relevance <m>r \\in \\mathbb{R}^{V}</m>; zero baseline; <m>N_r = 5</m> random orders"),
        ("ens", "gaps <m>\\Delta_{\\mathrm{del}}</m> and <m>\\Delta_{\\mathrm{ins}}</m>"),
        ("state", "<m>\\pi \\leftarrow</m> joints sorted by decreasing <m>r</m>"),
        ("for", "<m>s = 0, 1, \\dots, V</m>"),
        ("state", "<m>\\mathrm{del}_s \\leftarrow p_t</m> with joints <m>\\pi_1,\\dots,\\pi_s</m> set to zero"),
        ("state", "<m>\\mathrm{ins}_s \\leftarrow p_t</m> with joints <m>\\pi_{s+1},\\dots,\\pi_V</m> set to zero"),
        ("endfor", ""),
        ("for", "each of <m>N_r</m> random permutations"),
        ("state", "compute the random deletion and insertion curves in the same way"),
        ("endfor", ""),
        ("state", "average the random curves to <m>\\mathrm{del}^{\\mathrm{rand}}</m> and <m>\\mathrm{ins}^{\\mathrm{rand}}</m>"),
        ("return", "<m>\\Delta_{\\mathrm{del}} = A(\\mathrm{del}^{\\mathrm{rand}}) - A(\\mathrm{del})</m>, <m>\\Delta_{\\mathrm{ins}} = A(\\mathrm{ins}) - A(\\mathrm{ins}^{\\mathrm{rand}})</m>"),
    ])))
add(("eq", "eq:gaps", r"\Delta_{\mathrm{del}} = A_{\mathrm{rand}}^{\mathrm{del}} - A_{\mathrm{rel}}^{\mathrm{del}},\qquad \Delta_{\mathrm{ins}} = A_{\mathrm{rel}}^{\mathrm{ins}} - A_{\mathrm{rand}}^{\mathrm{ins}},",
     "<span class=\"m\">&Delta;<sub>del</sub> = A<sup>del</sup><sub>rand</sub> &minus; A<sup>del</sup><sub>rel</sub>, &nbsp;&Delta;<sub>ins</sub> = A<sup>ins</sup><sub>rel</sub> &minus; A<sup>ins</sup><sub>rand</sub></span>"))
add(("p", "so positive values mean the named joints matter more than random ones. The run, not the explanation, is the unit of analysis, because explanations from one model are not independent."))

add(("sub", "Statistical Analysis and Pre-Registration", "sub:stats"))
add(("p", "Intervals come from a paired clip-level bootstrap[@efron] (2,000 resamples, [#alg:boot]). Clips are resampled once and the same indices are used for every seed and arm, because seeds re-score the same clips and are not extra data. "
          "A treatment “reduces false alarms” only if (1) the 95% interval of the seed-averaged paired difference lies entirely below zero, "
          "(2) the treatment has fewer false alarms in at least three of four seeds, and (3) its false-alarm rate is lower at matched recall (0.60 and 0.65, read from the pooled out-of-fold operating curve). "
          "“Improves recall” uses the same test without (3). Failure of a rule is reported as “not shown”, which is not a finding of no effect. Seed 0 generated the hypotheses and is descriptive only. "
          "No multiplicity correction is applied. Faithfulness intervals use a run-level bootstrap (10,000 resamples) with runs weighted by the number of explanations they scored."))
add(("algo", dict(
    label="alg:boot", caption="Paired bootstrap and the pre-registered decision rule for “reduces false alarms”.",
    lines=[
        ("req", "per-clip false-alarm counts of treatment <m>X</m> and reference for seeds <m>s=1..S</m> (<m>S=4</m>); matched-recall rates; <m>B=2000</m>"),
        ("ens", "verdict: “reduces false alarms” or “not shown”"),
        ("for", "<m>b = 1, \\dots, B</m>"),
        ("state", "draw clip indices <m>I_b</m> with replacement, shared by all seeds and arms"),
        ("state", "<m>\\delta_b \\leftarrow \\frac{1}{S}\\sum_s \\big(\\mathrm{FA}^{X}_s(I_b) - \\mathrm{FA}^{\\mathrm{ref}}_s(I_b)\\big)</m>"),
        ("endfor", ""),
        ("state", "<m>c_1 \\leftarrow</m> the 97.5th percentile of <m>\\delta_b</m> is below zero"),
        ("state", "<m>c_2 \\leftarrow X</m> has fewer false alarms than the reference in at least 3 of 4 seeds"),
        ("state", "<m>c_3 \\leftarrow X</m> has a lower false-alarm rate at recall 0.60 and 0.65 wherever both curves reach them"),
        ("if", "<m>c_1 \\wedge c_2 \\wedge c_3</m>"),
        ("return", "“reduces false alarms”"),
        ("endif", ""),
        ("return", "“not shown”"),
    ])))
add(("p", "The decision rules for the confirmation grid and for the held-out-domain protocols were written before the runs and extended only by dated addenda. "
          "Version-control timestamps place the confirmation plan 11 s before the first confirmatory run and the held-out-domain plan four days before its first run. "
          "The first execution of the held-out-domain runs was invalid, because every run had tested on ordinary fold 0. This was detected because recall was identical across protocols with different test sets; "
          "none of those numbers is used, a verifier now checks that each run tested only on its held-out domain, and the experiment was rerun. "
          "The UR Fall seed 0–2 grid predates the confirmation plan, so the interval reported for its headline contrast (Section [#sec:results]) is exploratory."))
add(("sub", "Reproducibility and Availability", "sub:repro"))
add(("p", "Every run saves its resolved configuration, environment record (library versions, git commit), split, history, checkpoint and per-clip outcomes. "
          "A verification script (<code>scripts/verify_results.py</code>) recomputes the reported figures from the per-run totals, a different source from the per-clip analysis that produced them, and checks that all planned runs exist; "
          "all 27 checks pass. Code, configurations and the pre-registration files are available at <code>github.com/adityaayushman/FBimpact</code>. "
          "The datasets are public and must be obtained from their authors; the derived skeleton caches are not redistributed."))

# ============================================================================================ IV. Results
add(("sec", "Results", "sec:results"))
add(("sub", "Ablations on UR Fall", "sub:urfall"))
add(("p", "[#tab:urfall] and [#fig:urfall] show the six-variant ablation on UR Fall (30 falls, five folds). "
          "Removing the pre-impact time-weighted loss raised pooled recall from 0.633 to 0.767 (paired bootstrap difference +0.133, 95% CI +0.067 to +0.211) and left lead time unchanged (0.491 s against 0.489 s). "
          "The direction held in all three seeds (0.600 against 0.800, 0.733 against 0.800, 0.567 against 0.700). "
          "The plain baseline exceeded the full model by +0.078 (95% CI −0.022 to +0.189), higher in two of three seeds. "
          "Removing temporal modelling reduced recall to 0.333 and lead time to 0.336 s, and removing velocity features reduced recall to 0.533, so anticipation relies on motion over the window. "
          "These last two variants have a single seed each and are read as large effects rather than precise values. "
          "The baseline alone ranged from 0.600 to 0.833 across seeds, comparable to most between-variant gaps, and a single-seed run of the grid had first placed the baseline ahead of every variant. "
          "UR Fall holds only 5.8 minutes of normal activity, so one alarm moves a single fold’s false-alarm rate by about 52 per hour and false-alarm rates are not interpreted on this dataset."))
add(("table", dict(
    label="tab:urfall", wide=True, spec=r"p{0.30\textwidth}cccc",
    caption="UR Fall ablations (30 falls, five folds). Rows with 15 runs have three seeds; rows with 5 runs have one seed. Recall is pooled; lead time is the mean of per-run means. Intervals are exploratory paired bootstrap intervals.",
    header=["Variant", "Runs", "Recall", "Δ vs full model (95% CI)", "Lead time (s)"],
    rows=[
        ["Without pre-impact loss (<m>\\lambda</m> = 0)", "15", "0.767", "+0.133 (+0.067, +0.211)", "0.491"],
        ["Without grounding module", "5", "0.767", "n/a (one seed)", "0.531"],
        ["Baseline ST-GCN (no attention, <m>\\lambda</m> = 0)", "15", "0.711", "+0.078 (−0.022, +0.189)", "0.523"],
        ["<b>Full model (<m>\\lambda</m> = 1.5)</b>", "15", "<b>0.633</b>", "0", "0.489"],
        ["Without velocity features", "5", "0.533", "−0.100 (one seed)", "0.477"],
        ["Without temporal modelling", "5", "0.333", "−0.300 (one seed)", "0.336"],
    ])))
add(("fig", dict(label="fig:urfall", file="fig2_urfall", wide=False,
                 caption="UR Fall recall by variant. Grey circles are individual seeds and diamonds the pooled recall.")))
add(("sub", "Pre-Registered Confirmation on the Combined Benchmark", "sub:confirm"))
add(("p", "The hypothesis that the 14-channel features and four-phase supervision reduce false alarms came from one seed, on which the false-alarm count fell from 135 to 100 to 67 as arms A, B and C were introduced. "
          "It was therefore tested on four fresh seeds (80 runs: four arms, seeds 1 to 4, five folds). "
          "The arms differ in one component each: A uses the four channels <m>(x, y, v_x, v_y)</m>, B the 14-channel features, C adds the four-phase head with the consistency term to B, and D keeps the phase head but drops the consistency term. "
          "[#tab:arms] gives the arm means and [#tab:contrasts] the paired contrasts."))
add(("table", dict(
    label="tab:arms", wide=False, spec=r"p{0.34\columnwidth}cccc",
    caption="Combined benchmark, seeds 1–4 (mean ± sd over seeds; 129 falls, 26.4 min of normal activity). “Per h” is the mean count divided by 0.44 h.",
    header=["Arm", "Recall", "False alarms", "Per h", "Lead (s)"],
    rows=[
        ["A: four channels", "0.632±0.057", "108.3±18.2", "246", "0.426"],
        ["B: 14 channels", "0.653±0.053", "100.8±25.2", "229", "0.433"],
        ["C: B + phase head, consistency", "0.643±0.019", "83.5±4.7", "190", "0.435"],
        ["D: phase head, no consistency", "0.680±0.033", "77.3±11.6", "176", "0.410"],
    ])))
add(("table", dict(
    label="tab:contrasts", wide=True, spec=r"p{0.28\textwidth}cccc",
    caption="Paired contrasts under the pre-registered rule. Negative Δ false alarms favour the first-named arm.",
    header=["Contrast", "Δ false alarms (95% CI)", "Seeds with fewer", "Δ recall (95% CI)", "Verdict"],
    rows=[
        ["B − A: features", "−7.5 (−18.5, +4.3)", "2/4", "+0.021 (−0.019, +0.064)", "Not shown"],
        ["C − B: phase supervision", "−17.3 (−28.8, −6.8)", "2/4", "−0.010 (−0.056, +0.037)", "Not shown"],
        ["D − B: head and loss only", "−23.5 (−34.5, −13.5)", "2/4", "+0.027 (−0.023, +0.076)", "Not shown"],
        ["C − D: consistency coupling", "+6.3 (+2.0, +10.5)", "1/4", "−0.037 (−0.066, −0.008)", "Not shown"],
    ])))
add(("p", "Four-phase supervision (C) lowered the seed-averaged false-alarm count from 100.8 to 83.5 without a recall cost, and its interval lay below zero, satisfying the first condition. "
          "However, C produced fewer false alarms than B in only two of four seeds ([#fig:confirm]a), where the rule requires three. "
          "The aggregate reduction is driven largely by one unfavourable baseline seed (133 false alarms in seed 2); in the other seeds the two arms are close or B is ahead. "
          "The same pattern holds for D against B, the features contrast has an interval that includes zero, and the consistency term increased false alarms (+6.3) and lowered recall (−0.037) with intervals that exclude zero, "
          "an exploratory observation that this term may be harmful. Under the pre-registered rule every contrast is “not shown”, and the earlier drop from 135 to 67 was largely seed variation."))
add(("sub", "Operating Points at Matched Recall", "sub:matched"))
add(("p", "Moving along the operating curve reduces false alarms substantially ([#fig:confirm]b). At recall 0.60 the four arms sit at about 35 to 77 false alarms per hour (seed ranges), against 176 to 246 per hour at the default operating point. "
          "This is a secondary robustness check, because a threshold chosen on test scores is what the validation-frozen operating point exists to avoid. "
          "Descriptively, arms B–D lie below A (means 49 to 57 against 71 per hour at recall 0.60), but this was not confirmed by the pre-registered test. "
          "Even the best operating points correspond to tens of false alarms per hour of normal activity."))
add(("fig", dict(label="fig:confirm", file="fig3_confirmation", wide=True,
                 caption="Confirmation grid. (a) False-alarm counts per seed for arm B and arm C; C is lower in seeds 2 and 4 only. (b) False alarms per hour at fixed recall for the four arms (mean and range over seeds 1–4).")))
add(("sub", "Held-Out Domains", "sub:heldout"))
add(("p", "[#fig:heldout] and [#tab:heldout] show recall when a room or dataset was never seen in training. Recall depended strongly on the domain and moved in both directions. "
          "It fell by 0.23 to 0.39 for UR Fall to Le2i and for leaving the Coffee room out, and rose by 0.22 to 0.29 for Le2i to UR Fall and for leaving the Home scene out, relative to 0.64 in-domain. "
          "The highest recall (Le2i to UR Fall) came with the highest false-alarm rates (650 and 664 per hour). "
          "Seed-averaged frame-level AUC stayed between 0.75 and 0.99 in every cell, so the score kept most of its ranking ability while recall at the frozen threshold varied widely; "
          "whether this reflects threshold miscalibration under domain shift was not tested. The four-phase arm showed no consistent difference from the control: higher recall in two protocols, lower in two, "
          "and one seed collapsed to recall 0.03 on UR Fall to Le2i, where training used only 22 falls. These runs describe generalisation only and carry no information about the four-phase hypothesis."))
add(("fig", dict(label="fig:heldout", file="fig4_heldout", wide=False,
                 caption="Held-out-domain recall for each protocol, arm and seed. The dashed line is the in-domain recall of the control arm (seeds 1–3).")))
add(("table", dict(
    label="tab:heldout", wide=True, spec=r"lcccc",
    caption="Held-out-domain results (three seeds per cell). The last two columns give control / four-phase. One false alarm corresponds to 7–15 per hour on these test sets, so small differences in false-alarm rate are not interpretable.",
    header=["Protocol", "Control recall [range]", "Four-phase recall [range]", "Δ vs in-domain", "False alarms per hour"],
    rows=[
        ["UR Fall → Le2i", "0.411 [0.374, 0.455]", "0.253 [0.030, 0.364]", "−0.230 / −0.390", "148 / 71"],
        ["Le2i → UR Fall", "0.933 [0.867, 0.967]", "0.911 [0.867, 0.967]", "+0.292 / +0.268", "650 / 664"],
        ["Leave Coffee room out", "0.349 [0.339, 0.355]", "0.398 [0.290, 0.500]", "−0.292 / −0.245", "97 / 196"],
        ["Leave Home out", "0.865 [0.838, 0.892]", "0.892 [0.865, 0.919]", "+0.224 / +0.249", "142 / 100"],
    ])))
add(("sub", "Explanation Faithfulness", "sub:faithres"))
add(("p", "Across 90 real-data runs (677 explanations) both gaps were reliably positive ([#tab:faith], [#fig:faith]): deletion +0.043 (95% CI +0.012 to +0.074) and insertion +0.072 (+0.043 to +0.098), positive in 69% and 66% of runs. "
          "The effect is small and not uniform. It is largest for the combined-benchmark models and weak on UR Fall, and the point estimates for the Le2i baseline are negative with intervals that include zero. "
          "The pooled figures exclude 18 runs on the procedurally generated fixture; an earlier version of the pooling included them (+0.041, +0.069), a flaw that was found and corrected without changing the conclusion. "
          "These results neither confirm nor refute the general claim that attention weights are unreliable explanations[@jain]; "
          "they show that, for this architecture, joint attention carries a small but reliable amount of information about which joints the score depends on. The pooling was decided after the runs and is exploratory."))
add(("table", dict(
    label="tab:faith", wide=True, spec=r"lccrr",
    caption="Faithfulness gaps (attention relevance, zero baseline). Positive means better than a random joint order. Intervals resample runs.",
    header=["Models", "Runs", "Expl.", "Deletion gap (95% CI)", "Insertion gap (95% CI)"],
    rows=[
        ["<b>All real-data runs, pooled</b>", "90", "677", "+0.043 (+0.012, +0.074)", "+0.072 (+0.043, +0.098)"],
        ["Combined, B (14 channels)", "5", "88", "+0.072 (+0.030, +0.131)", "+0.114 (+0.081, +0.155)"],
        ["Combined, C (four-phase)", "5", "91", "+0.129 (+0.073, +0.192)", "+0.163 (+0.100, +0.216)"],
        ["Le2i, full model", "5", "64", "+0.062 (+0.002, +0.118)", "+0.098 (+0.041, +0.141)"],
        ["Le2i, baseline", "5", "67", "−0.116 (−0.252, +0.051)", "−0.075 (−0.176, +0.036)"],
        ["UR Fall, full model", "15", "57", "+0.051 (+0.003, +0.104)", "+0.021 (−0.034, +0.084)"],
        ["UR Fall, baseline", "15", "64", "+0.049 (+0.004, +0.088)", "+0.087 (+0.037, +0.136)"],
    ])))
add(("fig", dict(label="fig:faith", file="fig5_faithfulness", wide=True,
                 caption="Faithfulness gaps with 95% intervals over runs (number of runs in parentheses). The first row pools all real-data runs.")))
add(("sub", "Computational Cost and Verification", "sub:cost"))
add(("p", "On an RTX 4050 laptop GPU the base model runs at 16.3 ms per frame (61 frames per second) on GPU and 35.0 ms per frame (28.6 frames per second) on CPU at batch size 1, excluding pose estimation, which would dominate in deployment. "
          "The code base has more than 210 automated tests, including one that perturbs future frames and asserts that the current score does not change (causality), and one that asserts subject-disjoint splits where subject identity exists."))

# ============================================================================================ V. Discussion
add(("sec", "Discussion", "sec:discussion"))
add(("sub", "Principal Findings", "sub:principal"))
add(("p", "On RQ1, a causal skeleton model gives 0.41 to 0.49 s of mean lead time at 63 to 68% recall under clip-independent cross-validation, "
          "which is of the same order as wearable pre-impact detectors[@kfall][@prefallkd] and as the lead used in the camera study[@jesudhas]. "
          "The false-alarm rate is the dominant barrier: 176 to 246 per hour at the default operating point and still tens per hour at the best operating points. "
          "On RQ2, the named joints beat a random order by a small, reliable margin. On RQ3, the time-weighted pre-impact loss lowered recall on UR Fall in all three seeds and bought no lead time, "
          "and four-phase supervision did not meet the pre-registered criterion. On RQ4, in-domain recall is an unreliable guide to a new setting, in either direction."))
add(("sub", "Comparison with Prior Work", "sub:compare"))
add(("p", "[#tab:compare] sets published evidence against the present findings. The comparison is qualitative: the studies differ in task, data and split ([#tab:related])."))
add(("table", dict(
    label="tab:compare", wide=True, spec=r"p{0.30\textwidth}p{0.30\textwidth}p{0.34\textwidth}",
    caption="Existing evidence set against the findings of this work.",
    header=["Existing evidence or expectation", "Finding of this work", "Reading"],
    rows=[
        ["Skeleton ST-GCN models reach about 99% accuracy on UR Fall and Le2i[@shin].", "Streaming anticipation with a frozen threshold: pooled recall 0.63–0.77 (UR Fall, three-seed variants) and 0.63–0.68 (combined), at 176–246 false alarms per hour.", "Task and metric, not architecture, determine apparent performance. A 99% accuracy on fall versus non-fall (with 86.67% fall-class sensitivity in the same paper) does not transfer to anticipation under an alarm-rate criterion."],
        ["Pre-impact lead times are about 0.33–0.55 s for wearables[@kfall][@prefallkd] and 0.5 s in the camera study[@jesudhas].", "Realised mean lead time 0.41–0.49 s from skeletons, as the interval between a streaming warning and impact on warned falls.", "Similar magnitude. Lead is an outcome of a streaming detector here, and a fixed property of the input window in [@jesudhas]."],
        ["Fall-class recall of 0.72 at a 0.5 s lead with a random split[@jesudhas].", "Pooled recall 0.63–0.68 with a 0.4–0.5 s realised lead on clip-independent folds.", "Comparable in size, not in construction: the present protocol also charges false alarms per hour of normal activity."],
        ["Reported results rely on random or unspecified splits[@jesudhas][@shin].", "Held-out-domain recall 0.03–0.97 against 0.64 in-domain.", "Within-distribution results are a weak guide to a new room or camera; held-out-domain testing should be routine."],
        ["Attention weights may not be faithful explanations[@jain].", "Attention relevance exceeds a random order by +0.043 (deletion) and +0.072 (insertion).", "A small, reliable effect for this model, consistent with the view that the answer depends on the test used[@wiegreffe]."],
        ["A pre-impact objective and phase structure should help anticipation (design hypothesis).", "The pre-impact loss lowered UR Fall recall; phase supervision was not shown to reduce false alarms.", "Not supported. Null results are reported with the same prominence as positive ones."],
    ])))
add(("sub", "Why the Design Ideas May Have Failed", "sub:why"))
add(("p", "We did not test the causes, so the following are hypotheses. The time weighting favours earlier frames inside a 0.68 s window, but the window itself bounds the achievable lead, and the weighting shifts probability mass away from the later frames at which most warnings are confirmed, "
          "which would lower recall without lengthening lead. The phase labels derive from annotation proxies (a transition state on UR Fall, a fall-start frame on Le2i) whose timing is imprecise, "
          "and the consistency term couples two heads to a target that is itself uncertain."))
add(("sub", "Implications for Evaluation Practice", "sub:practice"))
add(("enum", [
    "Report the alarm rate per hour of normal activity and charge warnings issued before the imminent window.",
    "Treat lead time as an outcome measured on warned falls, with the operating point frozen on validation data.",
    "Include a held-out-domain test; in-domain recall can overstate or understate performance in a new place.",
    "Use several seeds and paired, pre-specified comparisons before accepting a design change. Two apparent improvements here (135 to 67 false alarms; the baseline’s superiority) were largely seed variation and were overturned by multi-seed testing.",
]))
add(("sub", "Limitations and Threats to Validity", "sub:limits"))
add(("p", "<i>Construct validity.</i> Both datasets contain acted falls by young volunteers, not recorded falls of older adults, and the impact frame is an annotation proxy. "
          "<i>Independence.</i> Splits are independent by clip, not by subject; a held-out scene removes the room and camera but not necessarily the people. "
          "<i>Sample size.</i> UR Fall has 30 falls and 5.8 minutes of normal activity, with per-fold recall from 0.12 to 1.00; held-out test sets hold 4.0 to 8.6 minutes of normal activity, so one false alarm moves the rate by 7 to 15 per hour. "
          "<i>Evidence strength.</i> Three UR Fall ablation variants have a single seed, the confirmation uses four seeds, and the four contrasts and two metrics were analysed without multiplicity correction. "
          "<i>Offline preprocessing.</i> Interpolation across dropped joints and scaling by the clip’s median torso length use frames after the current one, which a streaming system cannot. "
          "On a synthetic fixture the mean drift between offline and streaming normalisation was small (0.001 in position, 0.032 in velocity), but individual frames around a dropout differed sharply (worst case 11.98 in normalised units); "
          "the effect on the real data was not quantified, so offline results should be read as mild upper bounds on live behaviour. "
          "<i>Measurement.</i> One frozen pose estimator was used, and skeletons were usable in 80 to 85% of frames. "
          "<i>Scope.</i> The pre-impact loss was tested on UR Fall only, explanation methods other than attention were not compared, and the faithfulness pooling was decided after the runs."))
add(("sub", "Future Work", "sub:future"))
add(("p", "Three extensions follow directly: repeating the pre-impact-loss test on the combined benchmark with matched hardware and seeds, comparing gradient- and occlusion-based relevance with attention on the same checkpoints, "
          "and evaluating on data with actor labels so that splits can be subject-independent."))

# ============================================================================================ VI. Conclusion
add(("sec", "Conclusion", "sec:conclusion"))
add(("p", "A causal skeleton model can warn roughly 0.4 to 0.5 s before impact and detect about two thirds of acted falls, but at a false-alarm rate that is far too high for deployment, "
          "and with recall that varies from 0.03 to 0.97 across held-out rooms and datasets. The two design ideas intended to improve anticipation, a time-weighted pre-impact loss and four-phase supervision, "
          "were not supported by pre-registered tests. The joints named by the model’s attention carry a small, reliable signal about what drives its score. "
          "The methodological result matters as much as the model: pre-registered decision rules, multi-seed paired testing and held-out-domain evaluation changed three early conclusions, "
          "and the final figures are recomputed from saved runs by an automated check. We recommend that camera-based pre-impact detection be reported with an alarm-rate metric and a held-out-domain test."))

CSS = """
@page { size: letter; margin: 0.75in 0.625in 0.95in 0.625in; @bottom-center { content: counter(page); font: 8pt "Times New Roman", serif; } }
* { box-sizing: border-box; }
body { font: 10pt/1.2 "Times New Roman", Times, serif; color: #000; margin: 0; }
header { text-align: center; margin-bottom: 14pt; }
h1 { font: 24pt/1.12 "Times New Roman", serif; font-weight: normal; margin: 0 0 12pt; }
.a { font-size: 11pt; margin-bottom: 4pt; }
.au { display: flex; justify-content: center; gap: 0.5in; margin-bottom: 10pt; }
.au-block { font-size: 11pt; line-height: 1.3; }
.an { margin-bottom: 2pt; }
.aa { font-size: 10pt; }
.aff { font-size: 9pt; font-style: italic; }
main { column-count: 2; column-gap: 0.25in; text-align: justify; hyphens: auto; }
.abs { font-size: 9pt; font-weight: bold; margin: 0 0 6pt; }
p { margin: 0 0 0; text-indent: 0.2in; }
p + p, .eq + p { text-indent: 0.2in; }
h2 { font: 10pt "Times New Roman", serif; font-weight: normal; text-transform: uppercase; text-align: center; margin: 11pt 0 5pt; break-after: avoid; letter-spacing: 0.02em; }
h3 { font: italic 10pt "Times New Roman", serif; margin: 7pt 0 3pt; break-after: avoid; text-align: left; }
ol { margin: 3pt 0 3pt 0; padding-left: 0.28in; }
li { margin: 0 0 2pt; }
.eq { display: flex; align-items: baseline; justify-content: space-between; margin: 5pt 0; text-align: left; font-size: 9.5pt; }
.eqbody { flex: 1; text-align: center; }
.eqn { flex: 0 0 auto; padding-left: 6pt; }
.m { font-style: italic; } .m .rm { font-style: normal; } .m sub, .m sup { font-size: 70%; }
.m .cal { font-family: "Times New Roman", serif; font-style: italic; }
figure { margin: 6pt 0; break-inside: avoid; }
figure img { width: 100%; display: block; }
figcaption { font-size: 8pt; text-align: justify; margin-top: 3pt; }
.fig.wide, .tbl.wide, .alg { column-span: all; }
.fig.wide img { width: 100%; }
.tbl { margin: 6pt 0; break-inside: avoid; }
.tcap { font-size: 8pt; text-align: center; letter-spacing: 0.03em; }
.tttl { font-size: 8pt; text-align: center; text-transform: none; margin-bottom: 3pt; }
table { width: 100%; border-collapse: collapse; font-size: 8pt; line-height: 1.15; text-align: left; hyphens: manual; }
thead th { border-top: 0.8pt solid #000; border-bottom: 0.5pt solid #000; padding: 2pt 3pt; font-weight: bold; vertical-align: bottom; }
tbody td { padding: 1.6pt 3pt; vertical-align: top; }
tbody tr:last-child td { border-bottom: 0.8pt solid #000; }
.alg { margin: 6pt 0; break-inside: avoid; border-top: 0.8pt solid #000; border-bottom: 0.8pt solid #000; font-size: 8pt; line-height: 1.28; text-align: left; hyphens: manual; }
.algt { padding: 2pt 0; border-bottom: 0.5pt solid #000; margin-bottom: 2pt; }
.al { display: flex; padding: 0.3pt 0; }
.ln { flex: 0 0 12pt; text-align: right; padding-right: 5pt; color: #333; }
.at { flex: 1; }
.cm { color: #555; font-style: italic; margin-left: 4pt; }
.refh { margin-top: 12pt; }
.refs { padding-left: 0; list-style: none; font-size: 8pt; text-align: left; line-height: 1.15; }
.refs li { margin: 0 0 2.5pt; padding-left: 0.3in; text-indent: -0.3in; }
.rn { display: inline-block; width: 0.3in; text-indent: 0; }
a { color: inherit; text-decoration: none; }
code { font-family: "Courier New", monospace; font-size: 0.85em; }
"""


def main() -> int:
    for b in BLOCKS:
        if b[0] in ("sec", "sub") and not (len(b) > 2 and b[2]):
            raise SystemExit(f"every section needs a label: {b[1]}")
    doc = Doc(META, BLOCKS, REFS)
    (HERE / "fbimpact_ieee.tex").write_text(doc.tex(), encoding="utf-8")
    (HERE / "fbimpact_ieee.html").write_text(doc.html(CSS), encoding="utf-8")
    words = sum(len(b[1].split()) for b in BLOCKS if b[0] == "p")
    print(f"wrote paper/fbimpact_ieee.tex and .html | references: {len(doc.cite_order)} | prose words: {words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
