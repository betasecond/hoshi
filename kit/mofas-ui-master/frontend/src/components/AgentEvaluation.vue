<template>
  <div class="agent-evaluation">
    <h2>Evaluate Agents</h2>
    <button @click="evaluateAgents">Run Evaluation</button>
    <p>{{ result }}</p>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'AgentEvaluation',
  setup() {
    const result = ref('');

    const evaluateAgents = async () => {
      try {
        const response = await axios.get('/evaluation');
        result.value = response.data;
      } catch (error) {
        console.error('Evaluation failed:', error);
        result.value = 'Error occurred';
      }
    };

    return { result, evaluateAgents };
  },
});
</script>

<style scoped>
.agent-evaluation {
  margin-bottom: 20px;
}
</style>
