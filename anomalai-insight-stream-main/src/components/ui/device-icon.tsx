import { Camera, Smartphone, Glasses } from "lucide-react";

interface DeviceIconProps {
  deviceType: 'cctv' | 'phone' | 'glasses';
  className?: string;
}

export function DeviceIcon({ deviceType, className }: DeviceIconProps) {
  switch (deviceType) {
    case 'cctv':
      return <Camera className={className} />;
    case 'phone':
      return <Smartphone className={className} />;
    case 'glasses':
      return <Glasses className={className} />;
    default:
      return <Camera className={className} />;
  }
}