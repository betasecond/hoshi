import { createApp } from 'vue';
import App from './App.vue';
import router from './router';
import axios from 'axios';

// 设置后端地址
axios.defaults.baseURL = 'http://localhost:3000';

const app = createApp(App);
app.use(router);
app.mount('#app');
