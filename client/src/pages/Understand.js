import React from "react";
import "./Understand.css";
import { motion } from "framer-motion";

const Understand = () => {
  return (
    <div className="understand-container">
      <h2>Understanding the Dataset and Our Findings</h2>

      <motion.section
        className="dataset-description"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
      >
        <h3>About the Dataset</h3>
        <p>
          Dataset description goes here
        </p>
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
          The heatmap shows how different features in the dataset are correlated. 
          Values close to 1 indicate strong positive correlation, and values close to -1 indicate strong negative correlation. 
          For example, higher stress levels tend to correlate with poorer sleep quality.
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
          The confusion matrix visualizes the performance of our classification model, showing correct and incorrect predictions across categories. 
          Diagonal values represent accurate predictions, while off-diagonal values highlight misclassifications.
        </p>
      </motion.section>
    </div>
  );
};

export default Understand;
