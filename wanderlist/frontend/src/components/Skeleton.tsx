import { motion } from 'framer-motion';

interface SkeletonProps {
  className?: string;
}

const Skeleton: React.FC<SkeletonProps> = ({ className = '' }) => {
  return (
    <motion.div
      className={`bg-gray-200 rounded ${className}`}
      animate={{ opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.5, repeat: Infinity }}
    />
  );
};

export const CardSkeleton = () => {
  return (
    <div className="card overflow-hidden">
      <Skeleton className="aspect-video" />
      <div className="p-4">
        <Skeleton className="h-6 w-3/4 mb-2" />
        <Skeleton className="h-4 w-1/2 mb-2" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6 mt-1" />
      </div>
    </div>
  );
};

export default Skeleton;