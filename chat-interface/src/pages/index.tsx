import type { NextPage } from 'next';
import ChatInterface from '../components/ChatInterface';

const Home: NextPage = () => {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-800 flex flex-col">
      <ChatInterface />
    </div>
  );
};

export default Home;
