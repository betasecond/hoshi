<template>
  <div class="run-agent">
    <h2>Run Agent</h2>
    <input v-model="agentName" placeholder="Agent Name" />
    <button @click="runAgent">Run</button>
    <p>{{ result }}</p>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'RunAgent',
  setup() {
    const agentName = ref('reasoner');
    const result = ref('');

    const runAgent = async () => {
      try {
        const response = await axios.post('/run-agent', { agentName: agentName.value });
        result.value = response.data.result;
      } catch (error) {
        console.error('Run agent failed:', error);
        result.value = 'Error occurred';
      }
    };

    return { agentName, result, runAgent };
  },
});
</script>

<style scoped>
.run-agent {
  margin-bottom: 20px;
}
input {
  margin-right: 10px;
  padding: 5px;
}
</style>
