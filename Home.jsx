
import React, { useState } from 'react';
import axios from 'axios';

function Home() {
  const [datasetInfo, setDatasetInfo] = useState(null);

  const loadDataset = async () => {
    const res = await axios.get('http://localhost:8000/load-dataset');
    setDatasetInfo(res.data.info);
  };

  return (
    <div className="p-4">
      <h1 className="text-3xl font-bold mb-2">📰 Personalized News Finder</h1>
      <p className="mb-4 italic text-gray-600">AI-Powered News Classification and Recommendations</p>
      <button onClick={loadDataset} className="bg-blue-600 text-white px-4 py-2 rounded">Load Dataset</button>
      {datasetInfo && (
        <div className="mt-4">
          <h2 className="font-semibold">Dataset Summary:</h2>
          <pre>{JSON.stringify(datasetInfo, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default Home;
