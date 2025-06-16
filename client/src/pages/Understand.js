import React, { useEffect, useState } from "react";
import "./Understand.css";
import { motion } from "framer-motion";
import TestMentalHealth from "../components/TestMentalHealth";

const Understand = () => {
  const [testTrigger, setTestTrigger] = useState(false);

  useEffect(() => {
    const toUnderstand = localStorage.getItem("to-understand");
    if (toUnderstand === "true") {
      setTimeout(() => {
        setTestTrigger(true);
      }, 380);
      
    }
  }, []);

  //handle try mental health button click
  const handleTryClick = () => {
    setTestTrigger(!testTrigger); 
    localStorage.setItem("to-understand", "true");
  }

  return (
    <div className="understand-container">
      <TestMentalHealth trigger={testTrigger} setTrigger={setTestTrigger} />
      <h2>Understanding the Dataset and Our Findings</h2>

      <motion.section
        className="dataset-description"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h3>About the Dataset</h3>
        <p>
          The dataset used focuses on mental health assessments among students, 
          aiming to shed light on the various factors that may influence their psychological well-being. 
          It includes a diverse range of entries collected from anonymous sourcest. While the dataset offers valuable insights, it's important to note that like all data, it may contain errors or uncertainties and should not be assumed to be completely accurate. 
          Here is the link used for the dataset: https://www.kaggle.com/datasets/sonia22222/students-mental-health-assessments
        </p>
      </motion.section>


      <motion.section
        className="dataset-description"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h3>Machine Learning Algorithims used </h3>
        <p>
          To predict the likelihood of mental health risk among students,
          we employed two distinct machine learning approaches: Logistic Regression and an Artificial Neural Network (ANN). 
          These models were selected for their complementary strengths—Logistic Regression offers better interpretability, 
          while the ANN-style model provides flexibility and high performance on complex, structured data. The accuracy for Logisitic Regression is 87% while the accuracy for ANN is 55%. 
        </p>

        <p>
          In addition to model training, we implemented visualization tools to aid in interpretation (seen below).
          A correlation heatmap was generated to identify relationships between numerical features and mental health outcomes,
          and a confusion matrix was plotted to show how well the model distinguished between at-risk and not-at-risk students.
          We also calculated feature importance using permutation importance, highlighting which features—such as social support, 
          sleep quality, or academic stress—had the most significant influence on the model's decisions.
        </p>
        <button className="try-btn" onClick={handleTryClick}>Test your mental health</button>
      </motion.section>

      <motion.section
        className="insight-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <h3>Correlation Heatmap</h3>
        <img
          src="/assets/heatmap.png"
          alt="Correlation Heatmap"
          className="insight-image"
          loading="lazy"
        />
        <p>
          The heatmap shows how different features in the dataset are correlated with one another, including the target variable, mental health risk. 
          Values close to 1 indicate a strong positive correlation, meaning as one feature increases, so does the other. 
          Values close to -1 indicate a strong negative correlation, where one feature increases while the other decreases. 
          For example, we observed that higher levels of stress are positively correlated with increased mental health issues, while strong social support and good sleep quality may show negative correlations with mental health risk. 
          This visual representation helps identify which features may be influential predictors.
        </p>
      </motion.section>

      <motion.section
        className="insight-section"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.4 }}
      >
        <h3>Confusion Matrix</h3>
        <img
          src="/assets/confusion_matrix.png"
          alt="Confusion Matrix"
          className="insight-image"
          loading="lazy"
        />
       <p>
          The confusion matrix visualizes the performance of our classification model by comparing predicted versus actual outcomes. 
          The diagonal values represent correct predictions—true positives and true negatives—indicating how often the model classified a student correctly. 
          Off-diagonal values represent misclassifications, such as predicting “no risk” when the student was actually “at risk,” or vice versa. 
          This tool is especially useful in evaluating model performance across imbalanced classes, giving us deeper insight into where the model excels and where it needs improvement. 
          
      </p>
      </motion.section>
    </div>
  );
};

export default Understand;
