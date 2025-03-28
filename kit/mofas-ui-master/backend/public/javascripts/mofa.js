document.addEventListener('DOMContentLoaded', function() {
  // Helper function to display results or errors
  function displayResult(elementId, data, isError = false) {
    const element = document.getElementById(elementId);
    element.innerHTML = '';
    
    if (isError) {
      element.classList.add('error');
      element.textContent = `Error: ${data}`;
    } else {
      element.classList.remove('error');
      
      if (typeof data === 'object') {
        // If it's an array, format as list
        if (Array.isArray(data)) {
          const list = document.createElement('ul');
          data.forEach(item => {
            const listItem = document.createElement('li');
            listItem.textContent = item;
            list.appendChild(listItem);
          });
          element.appendChild(list);
        } else {
          // For objects, format as pre-formatted JSON
          const pre = document.createElement('pre');
          pre.textContent = JSON.stringify(data, null, 2);
          element.appendChild(pre);
        }
      } else {
        // For strings, just set the text content
        element.textContent = data;
      }
    }
  }

  // Helper function for API calls
  async function apiCall(url, method = 'GET', body = null) {
    try {
      const options = {
        method,
        headers: {
          'Content-Type': 'application/json'
        }
      };
      
      if (body) {
        options.body = JSON.stringify(body);
      }
      
      const response = await fetch(url, options);
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Unknown error');
      }
      
      return data;
    } catch (error) {
      throw error;
    }
  }

  // List Agents
  const listAgentsBtn = document.getElementById('list-agents-btn');
  if (listAgentsBtn) {
    listAgentsBtn.addEventListener('click', async function() {
      try {
        const resultsElem = document.getElementById('agent-list-results');
        resultsElem.textContent = 'Loading...';
        
        const data = await apiCall('/mofa/agent-list');
        displayResult('agent-list-results', data.agents);
      } catch (error) {
        displayResult('agent-list-results', error.message, true);
      }
    });
  }

  // Run Agent
  const runAgentBtn = document.getElementById('run-agent-btn');
  if (runAgentBtn) {
    runAgentBtn.addEventListener('click', async function() {
      try {
        const agentName = document.getElementById('run-agent-name').value.trim();
        
        if (!agentName) {
          throw new Error('Agent name is required');
        }
        
        const resultsElem = document.getElementById('run-agent-results');
        resultsElem.textContent = 'Running agent... This may take a while.';
        
        const data = await apiCall('/mofa/run', 'POST', { agentName });
        displayResult('run-agent-results', data.output);
      } catch (error) {
        displayResult('run-agent-results', error.message, true);
      }
    });
  }

  // Create New Agent
  const newAgentBtn = document.getElementById('new-agent-btn');
  if (newAgentBtn) {
    newAgentBtn.addEventListener('click', async function() {
      try {
        const agentName = document.getElementById('new-agent-name').value.trim();
        const version = document.getElementById('new-agent-version').value.trim();
        const output = document.getElementById('new-agent-output').value.trim();
        const authors = document.getElementById('new-agent-authors').value.trim();
        
        if (!agentName) {
          throw new Error('Agent name is required');
        }
        
        const resultsElem = document.getElementById('new-agent-results');
        resultsElem.textContent = 'Creating new agent... This may take a while.';
        
        const data = await apiCall('/mofa/new-agent', 'POST', { 
          agentName, 
          version: version || undefined, 
          output: output || undefined, 
          authors: authors || undefined 
        });
        
        displayResult('new-agent-results', data.output);
      } catch (error) {
        displayResult('new-agent-results', error.message, true);
      }
    });
  }

  // Run Evaluation
  const evaluationBtn = document.getElementById('evaluation-btn');
  if (evaluationBtn) {
    evaluationBtn.addEventListener('click', async function() {
      try {
        const resultsElem = document.getElementById('evaluation-results');
        resultsElem.textContent = 'Running evaluation... This may take a while.';
        
        const data = await apiCall('/mofa/evaluation', 'POST');
        displayResult('evaluation-results', data.output);
      } catch (error) {
        displayResult('evaluation-results', error.message, true);
      }
    });
  }

  // Agent Evaluation API
  const agentEvaluationApiBtn = document.getElementById('agent-evaluation-api-btn');
  if (agentEvaluationApiBtn) {
    agentEvaluationApiBtn.addEventListener('click', async function() {
      try {
        const primaryData = document.getElementById('primary-data').value.trim();
        const secondData = document.getElementById('second-data').value.trim();
        const comparisonDataTask = document.getElementById('comparison-data').value.trim();
        
        if (!primaryData || !secondData || !comparisonDataTask) {
          throw new Error('Primary data, secondary data, and comparison data task are required');
        }
        
        const resultsElem = document.getElementById('agent-evaluation-api-results');
        resultsElem.textContent = 'Running comparison... This may take a while.';
        
        const data = await apiCall('/mofa/agent-evaluation-api', 'POST', { 
          primaryData, 
          secondData, 
          comparisonDataTask 
        });
        
        displayResult('agent-evaluation-api-results', data.output);
      } catch (error) {
        displayResult('agent-evaluation-api-results', error.message, true);
      }
    });
  }

  // Run Custom Agent
  const runCustomAgentBtn = document.getElementById('run-custom-agent-btn');
  if (runCustomAgentBtn) {
    runCustomAgentBtn.addEventListener('click', async function() {
      try {
        const agentConfigStr = document.getElementById('agent-config').value.trim();
        
        if (!agentConfigStr) {
          throw new Error('Agent configuration is required');
        }
        
        // Try to parse as JSON
        let agentConfig;
        try {
          agentConfig = JSON.parse(agentConfigStr);
        } catch (e) {
          throw new Error('Invalid JSON format for agent configuration');
        }
        
        const resultsElem = document.getElementById('run-custom-agent-results');
        resultsElem.textContent = 'Running custom agent... This may take a while.';
        
        const data = await apiCall('/mofa/run-custom-agent', 'POST', { agentConfig });
        displayResult('run-custom-agent-results', data.output);
      } catch (error) {
        displayResult('run-custom-agent-results', error.message, true);
      }
    });
  }
});