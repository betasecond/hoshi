<template>
  <div class="new-agent">
    <h2>Create New Agent</h2>
    <input v-model="agentName" placeholder="Agent Name" />
    <input v-model="version" placeholder="Version (e.g., 0.0.1)" />
    <input v-model="authors" placeholder="Authors" />
    <button @click="createAgent">Create</button>
    <p>{{ result }}</p>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'NewAgent',
  setup() {
    const agentName = ref('');
    const version = ref('0.0.1');
    const authors = ref('Zonghuan Wu');
    const result = ref('');

    const createAgent = async () => {
      try {
        const response = await axios.post('/new-agent', {
          agentName: agentName.value,
          version: version.value,
          authors: authors.value,
        });
        result.value = response.data.message;
        agentName.value = ''; // 重置输入
      } catch (error) {
        console.error('Create agent failed:', error);
        result.value = 'Error occurred';
      }
    };

    return { agentName, version, authors, result, createAgent };
  },
});
</script>

<style scoped>
.new-agent {
  margin-bottom: 20px;
}
input {
  margin-right: 10px;
  padding: 5px;
  margin-bottom: 10px;
}
</style>
