import streamlit as st
import os
from experiment_pipeline.data_loader import load_global_feature_set
import pandas as pd

SAVED_EXPERIMENT_DIR = "saved_experiments/"
DATA_DIR = "data/streamlit/"

def st_demo_weather():
    st.header("Motivation")
    st.write("""
        The motivation behind a lot of this work was to understand how severe weather impacts bus ridership in NYC. Let’s see how the tools we developed can be used to gain insight into that question.
    """)

    st.header("Establishing a Baseline")
    st.write("""
        A sensible first step in assessing the extent to which weather conditions influence the number of people who ride the bus on a given day (in the case of our analysis, the B46 between 8/1 and 9/30/2021) is to establish a baseline predictive model trained in the absence of weather features (e.g., bus position, observation time, and certain timetable details). Then, we can compare the performance of our baseline against that of a substantially similar model trained on exactly the same data plus some additional weather features (e.g., precipitation, temperature, and humidity). If the augmented model performs better than our baseline, we can say that the inclusion of weather features improves our model’s ability to predict bus ridership. Conversely, if the augmented model performs inline with (or worse than) our baseline, we might begin to question if weather has anything to do with people’s decision to ride the bus or, at the very least, attempt to diagnose why our data (or representation thereof) did not lend itself to the prediction task.
    """)

    st.subheader("An Aside on the Prediction Task")
    st.write("""
        Just to we are all on the same page, the specific task we are training our model to perform is the prediction of the number of passengers on a specific vehicle at a specific time and place. To help illustrate that, here are the first five rows of the combined bus and weather training data.
    """)

    # TODO: train_bus_weather.head()

    st.subheader("Model Selection")
    st.subheader("Linear - possible to do sub-subheader?")
    st.write("""
        Our first attempt at establishing a baseline model was to train a first-degree, L1-regularized linear regressor on 6 of our 8 weeks of bus data, then test on the final 2 weeks. To evaluate performance, we looked primarily at mean absolute error (MAE) on basis of interpretability with respect to the prediction task at hand (i.e., how many people are currently on the bus). Overall, our linear model performed surprisingly well for a baseline, with an MAE of 7.8 on the test set, meaning that, on average, it was able to correctly predict occupancy to within +/- 8 people relative to the number of riders actually observed. When you stop and think about it, that’s not bad at all. In fact, you probably wouldn’t even notice whether there were 8 more or 8 fewer passengers on a given bus (particularly the larger articulated ones that serve the B46). Indeed, we can see our model does a pretty good job of capturing the ebb end flow of ridership over the course of a day (at least at a macro level).
    """)

    # TODO: Lasso.experiment_eval.plot_passenger_count_by_time_of_day (sum)

    st.write("""
        However, that’s not all we care about when evaluating the performance of a regression model. We also want to understand how well it captures the variance in the data. We can see below that our baseline struggles mightily here.
    """)

    # TODO: streamlit: Lasso.experiment_eval.plot_passenger_count_by_time_of_day (mean)
    
    st.write("""
        This is confirmed by an abysmal R^2 score of -0.02, meaning that as good as things were looking for us a minute ago, our baseline is actually a slightly worse model than simply predicting per stop passenger count averages learned on the training set, which achieves MAE and R^2 scores of 7.7 and 0.00, respectively.
        
        We can do better...
    """)
    
    st.subheader("Gradient Boosted Tree - possible to do sub-subheader?")
    st.write("""
        To address underfitting, we decided to go with a more expressive model class for our second attempt at establishing a baseline, namely XGBoost. Right away, we see a marked improvement in both average error and explanation of variance:
    """)

    # TODO: summary results df1... see doc
    # TODO: XGBoost.experiment_eval.plot_passenger_count_by_time_of_day (sum)
    # TODO: XGBoost.experiment_eval.plot_passenger_count_by_time_of_day (mean)

    st.write("""
        Although things look pretty good in terms of the underfitting problem we were trying to solve with XGBoost, when thinking about predicted versus ground truth passenger counts across the whole route, they break down a bit when you start drilling down to stop-specific predictions (which, although we didn’t show, were also an issue for our linear model):
    """)

    # TODO: XGBoost.experiment_eval.plot_passenger_count_by_time_of_day (mean, stop) – include dropdown with all the stops 

    st.write("""
        Admittedly, there are a lot of potential reasons that one might see this kind of behavior but since we believe it speaks more to higher-level decisions around problem formulation, data modeling, and training procedures than algorithm selection. So, for now we’ll press forward with XGBoost as our baseline and leave the discussion of our approach’s shortcomings for a later section.
    """)

    st.header("Adding Weather Features")
    st.write("""
        We can now train a new instance of our baseline model on bus and weather data and compare the results:
    """)

    # TODO: summary results df2... see doc

    st.write("""
        What we find, is that adding weather features actually diminishes our model’s predictive capacity—how disappointing!

        Now, if, hypothetically speaking, you were trying to predict bus occupancy using time, location, and weather data for a course project, you might consider, instead, shifting the focus of your work toward the development of an open-source repo that others can use to push the ball forward, for example. :P

        But let’s be good sports about it and try and gain some insight into what’s going on…

        A good place to start would be to look at the correlations that exist between the various features:
    """)
    
    # TODO: streamlit: plot_feature_correlation(subset=[bus, weather, and all features]) – dropdown

    st.write("""
        Interestingly, heat index and relative humidity are the two most highly-correlated features with passenger count.  While this implies that the inclusion of such features would improve the predictions of a linear model, our current baseline has learned non-linear relationships between the features that are more relevant to the prediction task than the linear relationships described in the correlation matrix above. Indeed, although adding weather features to our preliminary Lasso model, for instance, would see MAE and R^2 scores improve to 7.6 and 0.01, from 7.8 and -0.02, respectively, it still vastly underperforms the current XGBoost baseline.

        To (begin to) get a sense for the non-linear relationships learned by our baseline, we can inspect feature importance, which, in the context of XGBoost, is basically a measure of information gain:
    """)
    
    # TODO: XGBoost.experiment_eval. plot_feature_importance(ablate_features=False)

    st.write("""
        Going a step further, we can see how each the inclusion of each successive feature improves or diminishes model performance:
    """)

    # TODO: XGBoost.experiment_eval. plot_feature_importance(ablate_features=True)

    st.write("""
        The upshot is that not only are weather features of minimal importance to our baseline, it can achieve essentially peak performance using only just time and location!
    
        But why is this? Intuitively, weather should have some impact on bus ridership. To help gain some insight into what is going on, let’s look at our training and testing data:
    """)

    # TODO: XGBoost.experiment_eval.gt_pred_scatter(data=data_toggle=[‘train’, ‘test’], plot='datetime', errors=error_toggle=['large', ‘small’, ‘all’], n=n_toggle=[50, 100, 500, 1000])
    
    st.write("""
        Ignoring the dots for a moment (more on that in the section on error analysis), we can see that a big issue from an evaluation perspective is that there are no weather events in our testing data! How is a model that learns, for example, that fewer people take the bus on hot and humid days supposed to perform on a test set that doesn’t have any hot and humid days? 
        
        Overall, as with the low variance observed in our baseline’s stop-specific predictions, we view its inability to glean information related to the prediction task from weather features as more of a high-level issue than one of model selection. Moreover, we believe there exist significant room to improve the approach to this problem than the one outlined above.
    """)

    st.header("Error Analysis")
    st.write("""
        Hypothetically speaking, if you had a reasonably well-functioning model, one of the things you might want to do as part of your evaluation / fine-tuning of it is to inspect where it is making mistakes. 

        A first step might be to see how your model’s predictions compare to ground truth observations:
    """)

    # TODO: XGBoost. experiment_eval.gt_pred_scatter('test', errors='all', n=100, s=200) – all toggles except “s”…

    st.write("""
        But because time and space are dimensions to consider (i.e., date, time of day, bus stop, etc.), being able to disambiguate errors made along these dimensions is critical for model development
    """)

    # TODO: XGBoost. experiment_eval.gt_pred_scatter('test', plot='simple', errors='all', n=100, s=200, y_axis='gt', overlay_weather=False) – all the toggles…

    st.header("Parting Thoughts")
    st.write("""
        Though the weather result was clearly disappointing, we think there exists significant room to improve the methods outlined in this demo. 
        
        First of all, our model does not incorporate the inherently sequential nature of the data. The challenge with this kind of representation, however, is unlike conventional timeseries data, ours is made up of many shorter sequences (i.e., individual buses making stops along a route) that overlap one another and generally don’t individually span the entire time window you are looking at. Moreover, because our data is collected at fixed intervals that in many cases exceed the amount of time it takes a bus to travel between two stops, it is generally not possible to establish a 1:1 mapping between set of datapoints collected from an individual vehicle to the full sequence of stops it is scheduled to make (at least not directly from the data or without significant interpolation). 
        
        Second, we likely didn’t look at a long enough period of time to “see” enough weather events to learn anything meaningful. Additionally, the route we focused on, the B46 runs through the heart of Brooklyn, so commuters who rely on it arguably don’t have the kind of transit alternatives (and may therefore be less sensitive to weather conditions) as those who take the M103, for instance, which parallel to the 6 train along Lexington Ave. in Manhattan for a significant portion of its route. 
        
        Third, by formulating the task as predicting of the number of passengers on a specific vehicle at a specific time and place, we preclude the use of information about the state of the network elsewhere at inference time. A potentially interesting alternative to our problem formulation would be to think about the task as predicting the edge weights of a directed acyclic graph, where edge weights are passenger counts aggregated over some period of time.
    """)








    