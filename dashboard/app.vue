<!-- dashboard/app.vue -->

<template>
  <div>
    <h1>Taxi Fare Prediction and Anomaly Detection</h1>
    <div>
      <h2>Training</h2>
      <button @click="trainModels">Train Models</button>
    </div>
    <div>
      <h2>Prediction</h2>
      <input v-model="newData" placeholder="Enter new data as JSON" />
      <button @click="predictFare">Predict Fare</button>
      <p v-if="predictions.length">Predictions: {{ predictions }}</p>
    </div>
    <div>
      <h2>Anomaly Detection</h2>
      <input v-model="newData" placeholder="Enter new data as JSON" />
      <button @click="detectAnomalies">Detect Anomalies</button>
      <p v-if="anomalies.length">Anomalies: {{ anomalies }}</p>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'App',
  data() {
    return {
      newData: '',
      predictions: [],
      anomalies: []
    }
  },
  methods: {
    trainModels() {
      axios.post('/train')
        .then(response => {
          console.log(response.data.message)
        })
        .catch(error => {
          console.error(error)
        })
    },
    predictFare() {
      const data = JSON.parse(this.newData)
      axios.post('/predict', data)
        .then(response => {
          this.predictions = response.data.predictions
        })
        .catch(error => {
          console.error(error)
        })
    },
    detectAnomalies() {
      const data = JSON.parse(this.newData)
      axios.post('/detect_anomalies', data)
        .then(response => {
          this.anomalies = response.data.is_anomaly
        })
        .catch(error => {
          console.error(error)
        })
    }
  }
}
</script>