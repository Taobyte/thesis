# Non-Stationary Forecasting
Time series forecasting plays a critical role in understanding and predicting dynamic systems
across various domains, including finance, healthcare, and environmental monitoring. However,
accurately forecasting non-stationary signals—data characterized by changing statistical properties
such as mean, variance, or seasonality over time—remains a significant challenge. Non-stationary
signals often arise from complex, real-world events like abrupt market shifts, extreme weather
patterns, or evolving biological processes.

This project aims to investigate the effectiveness of current machine learning-based forecasting
models in predicting non-stationary events. By evaluating the performance of established methods
such as GPT-based approaches on real-world non-stationary datasets, this study seeks to identify
their strengths, limitations, and applicability. The findings will help determine whether these
models can reliably capture the patterns and abrupt transitions inherent to non-stationary signals,
or if advancements are needed to improve their predictive accuracy in such scenarios 

Current research heavily focusses on public datasets. For long-term forecasting the datasets ETT, Electricity, Weather, Exchange and ILI are frequently used for evaluating novel machine learning models. For short-term forecasting, many authors use the M4 dataset. You can download these datasets from this google drive [Datasets](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) provided by the authors from the TimesNet paper.




## Datasets



| Dataset     | Signal Type              | Variates                  | Activity Type                     | Frequency | 
|-------------|--------------------------|---------------------------|-----------------------------------| -------- |
| [WildPPG](https://arxiv.org/abs/2412.17540)    | PPG / HR                    | 1          | Accelerometer                  |25 Hz |
| [DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia)      | PPG / HR | 1            | Accelerometer   | 32 Hz|
| [IEEE](https://ieeexplore.ieee.org/document/6905737)         | PPG / HR           | 1            | Accelerometer  | 25 Hz|

## Complex Deep Learning Models 

| Model        | Description | Supports Short Seq | Link |
|--------------|-------------|---------| ----- |
| TimesNet     | Transformer-based model tailored for long-term time series forecasting. Excels in multivariate settings.| No | [GitHub](https://github.com/thuml/Time-Series-Library/) |
| SimpleTM    | Lightweight and simple transformer with minimal changes, offering strong performance on time series tasks. | Yes| [GitHub](https://github.com/vsingh-group/SimpleTM/) |
| ElasTST   | Patch-based transformer with rotary embeddings and elastic structure, effective for non-stationary series. | No| [GitHub](https://github.com/microsoft/ProbTS/tree/elastst) |
| Ada-MSHyper  | Adaptive multi-scale hypernetwork that dynamically generates model weights based on input resolution. | No | [GitHub](https://github.com/shangzongjiang/Ada-MSHyper) |
| GPT4TS      | Unified GPT-based model that performs multiple time series tasks without retraining ("one fits all"). | Yes | [GitHub](https://github.com/DAMO-DI-ML/)
Pattn | Simple baseline forecasting model based on patching and the transfomer architecture | Yes | [GitHub](https://github.com/BennyTMT/LLMsForTimeSeries/tree/main)  
TimeLLM |  LLM-based timeseries forecasting model using a reprogramming layer and using a frozen LLM as backbone. | Yes | [GitHub](https://github.com/ngruver/llmtime) |

# TODO ADD THIS INFORMATION TO THE README
\subsubsection{TimesNet}
The Timesnet model \cite{wu2023timesnettemporal2dvariationmodeling} works by stacking so-called TimesBlocks on top of each other. 
Each TimesBlock uses the Fast Fourier Transform to extract relevant periods from the multivariate timeseries input. Next, these timeseries are transformed into 2D tensors based on the $k$ periods extracted in the previous stage and fed into a parameter efficient inception block that uses 2D convolutions. Lastly, adaptive aggregation is used, where the $k$ amplitudes are used for weighting the corresponding series representation. For an overview, consider the Figure \ref{sota_timesnet_overview}.The authors claim that this architecture enables extracting multi-scale temporal 2D-variations simultaneously. 

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures/related_work/timesnet_architecture_overview.png}
    \caption{\textbf{TimesNet Architecture Overview}}
    \label{sota_timesnet_overview}
\end{figure}

\subsubsection{SimpleTM}
The SimpleTM model \cite{chen2025simpletm} is composed of three stages. In the first stage, the multivariate timeseries is tokenized by a learnable stationary wavelet transform. In the second stage, the authors then futrher process the tokens by introducing a geometric product attention, a concept derived from geometric algebra. In the last stage, the transformed token representation is transformed back to a 1D timeseries representation by now using a learnable inverse stationary wavelet transform. In summary, the authors propose to use a wavelet transform to extract multi-scale features and introduce the geometric attention mechanism. The reader can find an overview in Figure \ref{sota_simpletm_overview}. 


\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures/related_work/simpletm_architecture_overview.png}
    \caption{\textbf{SimpleTM Architecture Overview}}
    \label{sota_simpletm_overview}
\end{figure}


\subsubsection{AdaMSHyper}
AdaMSHyper \cite{shang2024adamshyperadaptivemultiscalehypergraph} stands for Adaptive Multi-Scale Hypergraph Transformer and is based on hypergraphs and graph neural networks. This architecture contains four modules \ref{sota_adamshyper_overview}. In the first module, multiple convolutions are used to transform the 1D timeseries into nodes in the hypergraph. This is done iteratively creating nodes that contain fine-grained and coarse-grained information. The second module is used to learn the hyperedges for each scale and nodes created in the first module. The module creates incidence matrices at each scale to model implicit groud-wise node interactions. The authors also implement a node an hyperedge constraint mechanism that clusters nodes with similar semantic information. The goal in the third module is to extract intra-scale and inter-scale interactions. For the inter-scale interactions, the authors propose a novel hypergraph convolution attention mechanism to capture group-wise interactions among nodes with similar semantic information. For the inter-scale interactions, they use plain attention mechanism with the learnt hyperedges. The last module uses a simple feed-forward network, that takes as input the concatenated node and hyperedge features and predicts the 1D timeseries.  

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures//related_work/adamshyper_architecture_overview.png}
    \caption{\textbf{AdaMSHyper Architecture Overview}}
    \label{sota_adamshyper_overview}
\end{figure}

%\subsubsection{PAttn}
\subsubsection{TimeXer}
The TimeXer model \cite{wang2024timexerempoweringtransformerstime} specifically design to distinguish between endogenous and exogenous variables. The model is again divide into four stages. In the beginning, the endogenous 1D timeseries is split up into non-overlapping patches with patch length $p > 1$. In contrast, the exogenous variables are not transformed into a patched representation, each channel is compressed into variat tokens. Since the two representations are not the same, the authors introduce a global token that bridges the gap between the endogenous and exogenous representations. TimeXer now adopts the transformer architecture in the following way. The first attention block is the standard self-attention mechanism for the endogenous patches. The second attention block is replaced by a cross attention block where the endogenous patches serve as the queries and the variat tokens as the keys. The resulting representation is then concatenated with the global token and fed into the feed-forward network. For a visual overview, we refer to Figure \ref{sota_timexer_overview}

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures//related_work/timexer_overview.png}
    \caption{\textbf{TimeXer Architecture Overview}}
    \label{sota_timexer_overview}
\end{figure}

\subsubsection{GPT4TS}
The authors from \cite{zhouOneFitsAll2023} propose to use a pretrained GPT2 language model for timeseries forecasting. The reader can find an overview of the architecture in Figure \ref{sota_gpt4ts_overview}. At the start, the timeseries is fed into an instance normalization layer, followed by transforming it into a patched representation. Next, an learnable input embedding layer is applied to transform the patched timeseries into the required dimension of the pretrained GPT2 model. The GPT2 model contains a stack of transformer blocks and positional embeddings. The authors freeze the multi-head attention mechanism and feed forward networks in the transformer blocks and only finetune the positional embeddings and layer norms in the transformer blocks.  


\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures//related_work/gpt4ts_architecture_overview.png}
    \caption{\textbf{GPT4TS Architecture Overview}}
    \label{sota_gpt4ts_overview}
\end{figure}


## Baselines
- Linear Models
- [Kalman Filters](https://www.quantstart.com/articles/State-Space-Models-and-the-Kalman-Filter/)
- [Gaussian Processes](https://gpytorch.ai/)
- [Bayesian Neural Networks](https://docs.pyro.ai/en/dev/contrib.bnn.html)
- [XGBoost](https://xgboost.readthedocs.io/en/stable/)

## Installation 
This project uses [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) for dependencies management. Use the following commands in your favourit shell. 

```
git clone https://github.com/eth-siplab-students/t-mt-2025-NonstationaryForecasting-ClemensKeusch.git

cd t-mt-2025-NonstationaryForecasting-ClemensKeusch
conda env create -f environment.yaml
conda activate thesis 
```

## Running Experiments 
This project uses [Hydra](https://hydra.cc/) and  [WandB](https://wandb.ai/)  for experiment tracking. In the following, I explain how to run the code locally or on a SLURM cluster (all models in this project were trained on the [Euler cluster](https://scicomp.ethz.ch/wiki/Euler)). Make sure to always activate the conda environment. 
``` 
conda activate thesis
```
Since we use the hydra library, you can configure all parameters for the dataset or model either with the command line or directly in the config file. The most important config file is `` config/config.yaml``. 
The first part of the ``config.yaml`` file contains 
```
defaults:
  - _self_
  - model: elastst
  - dataset: dalia
  - path: local
```
Here you can specify what model you want to train and which dataset should be used. The name of the model and dataset has to be the same as the corresponding yaml file in the ``config/model/`` and ``config/dataset/`` directory. 

As already hinted at above, you can also directly change the config parameters in the command line. For this use the following syntax.
```
python main.py model=simpletm dataset=wildppg seed=17
```
Even if the config file would contain the dalia dataset, the command above would override this information with the wildppg dataset. 

### Local 
For training and testing a model locally, you can use 
```
python main.py 
```

### SLURM cluster
For training models on the SLURM cluster, we use the [Submitit Launcher plugin](https://hydra.cc/docs/plugins/submitit_launcher/) for hydra. This plugin allows specifying the SLURM related parameters again in a config file and hence neatly integrates into the hydra framework. Hydra also allows sweeping over different parameter configurations. This makes running ablations studies efficient. Make sure to always use the flag ``--multirun``, even if you only run a single experiment. 

As an example consider the following command.

```
python main.py --multirun hydra/launcher=submitit_slurm look_back_window=3,4,5 prediction_window=1,2,3
```
This command sweeps over all lookback window and prediction window combinations (in this case 9 combinations) and spawns a job for each of them. 


### Hydra Optuna Sweeper 
On the 30th of April, I installed the hydra optuna sweeper plugin using 
```
pip install hydra-optuna-sweeper --upgrade
```
(see https://hydra.cc/docs/plugins/optuna_sweeper/).

This plugin does not natively support WandB. Thus, I added the `optuna.integration.WeightsAndBiasesCallback` to the `_imp.py` file and added three parameters to the sweeper, namely `use_wandb, metric_name, wandb_kwargs`. 
For now, I just copy the changes I made locally to the cluster, but later I may have to create a package to install it properly. 
