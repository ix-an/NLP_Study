<template>
	<view class="container">
		<view class="page-title">MyDeepSeek</view>
		
		<view class="button-group">
			<button class="main-btn" type="primary" @tap="gotoDeepseek">
				<text class="iconfont icon-comments"></text>
				和 DeepSeek 对话
			</button>
			
			<button class="func-btn" type="default" @tap="showMsg">
				<text class="iconfont icon-info"></text>
				显示提示消息
			</button>
			
			<button class="func-btn" type="default" @tap="showLoading">
				<text class="iconfont icon-loading"></text>
				显示数据加载
			</button>
			
			<button class="func-btn" type="warn" @tap="delSubmit">
				<text class="iconfont icon-trash"></text>
				删除
			</button>
		</view>
	</view>
</template>


<style>
	.container {
		display: flex;
		flex-direction: column;
		min-height: 100vh;
		padding: 20rpx;
		background-color: #f5f5f5;
	}
	
	.page-title {
		text-align: center;
		font-size: 36rpx;
		font-weight: bold;
		color: #333;
		margin: 60rpx 0 80rpx;
	}
	
	.button-group {
		display: flex;
		flex-direction: column;
		gap: 30rpx;
		padding: 20rpx;
	}
	
	.main-btn {
		height: 100rpx;
		line-height: 100rpx;
		font-size: 32rpx;
		border-radius: 50rpx;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 15rpx;
	}
	
	.func-btn {
		height: 90rpx;
		line-height: 90rpx;
		font-size: 28rpx;
		border-radius: 50rpx;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 15rpx;
	}
	
	/* 图标字体样式 */
	.iconfont {
		font-size: 30rpx;
	}
</style>


<script>
	export default {
		data() {
			return {
				// 可以在这里定义需要的数据
			}
		},
		onLoad() {
			// 页面加载时的初始化操作
			console.log('首页加载完成');
		},
		methods: {
			// 跳转到对话页面
			gotoDeepseek() {
				uni.switchTab({
					url: '/pages/chat/chat',
					fail: (err) => {
						console.error('跳转失败:', err);
						uni.showToast({
							title: '跳转失败',
							icon: 'none'
						});
					}
				})
			},
			
			// 显示提示消息
			showMsg() {
				uni.showToast({
					title: '请输入你的问题',
					icon: 'none',  // 不显示图标，纯文本
					duration: 2000
				})
			},
			
			// 显示加载提示，并3秒后自动关闭
			showLoading() {
				uni.showLoading({
					title: '加载中...',
					mask: true
				});
				
				// 3秒后自动关闭加载提示
				setTimeout(() => {
					uni.hideLoading();
				}, 3000);
			},
			
			// 删除确认对话框
			delSubmit() {
				uni.showModal({
					title: '确认删除',
					content: '是否确定要删除该内容？',
					confirmText: '删除',
					cancelText: '取消',
					success: (res) => {
						if (res.confirm) {
							// 这里可以添加实际的删除逻辑
							uni.showToast({
								title: '删除成功',
								icon: 'success',
								duration: 1500
							});
						}
					}
				})
			}
		}
	}
</script>
