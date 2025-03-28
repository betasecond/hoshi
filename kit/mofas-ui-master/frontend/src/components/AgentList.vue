<template>
  <div class="agent-list">
    <h2>Agent List</h2>
    <button @click="fetchAgents">Refresh</button>
    <ul>
      <li v-for="agent in agents" :key="agent">{{ agent }}</li>
    </ul>
  </div>
</template>

<script lang="ts">
import { defineComponent, ref } from 'vue';
import axios from 'axios';

export default defineComponent({
  name: 'AgentList',
  setup() {
    const agents = ref<string[]>([]);

    const fetchAgents = async () => {
      try {
        const response = await axios.get('/agent-list');
        agents.value = response.data.agents;
      } catch (error) {
        console.error('Failed to fetch agents:', error);
      }
    };

    fetchAgents(); // 初始化时加载

    return { agents, fetchAgents };
  },
});
</script>

<style scoped>
.agent-list {
  margin-bottom: 20px;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  padding: 5px 0;
}
</style>
