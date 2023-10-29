/*
 * Copyright 2009-2017 Alibaba Cloud All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>
#include <iostream>
#include <alibabacloud/oss/OssResult.h>
#include <alibabacloud/oss/Types.h>

namespace AlibabaCloud
{
namespace OSS
{
    class ALIBABACLOUD_OSS_EXPORT GetBucketRefererResult : public OssResult
    {
    public:
        GetBucketRefererResult();
        GetBucketRefererResult(const std::string& data);
        GetBucketRefererResult(const std::shared_ptr<std::iostream>& data);
        GetBucketRefererResult& operator=(const std::string& data);
        const AlibabaCloud::OSS::RefererList& RefererList() const { return refererList_;}
        bool AllowEmptyReferer() const { return allowEmptyReferer_; }
    private:
        AlibabaCloud::OSS::RefererList refererList_;
        bool allowEmptyReferer_;
    };
} 
}
